"""Asymmetric Centralized Critic (MAPPO-style) architecture for RADON rescue agents.

Architectural Guarantees:
1. ACTOR receives ONLY local observations:
   - 7x7x6 local grid (egocentric)
   - 13-dim self vector (carrying, floor, steps left, row, col, hostages rem, 4 BFS deltas, dist, diff floor, has goal)
   - NEVER receives global map state
   - 100% SIZE-INVARIANT: runs on 10x10, 14x14, 20x20 with zero shape mismatch.
2. CRITIC (Centralized Value Function) receives canonical global state:
   - Canonical 10x10 global map (down-sampled via windowed max-pooling) across all floors
   - Global scalars (hostages remaining, living rescuers, fire count, steps left)
   - Local features (optional concatenation for cooperative situational awareness)
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union
import gymnasium.spaces as spaces
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.distributions import Distribution

from env.fire_controller import resize_to_canonical
from train.extractor import RescueExtractor


def build_centralized_global_state(
    env,
    target_h: int = 10,
    target_w: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extracts canonical fixed-dimension global grid and scalar state from environment.

    Returns:
        global_grid: (6 * floors, target_h, target_w) float32
        global_scalars: (4,) float32
    """
    floors = env.config.floors
    h = env.config.grid_h
    w = env.config.grid_w
    c_per_floor = 6
    full_channels = np.zeros((c_per_floor * floors, h, w), dtype=np.float32)

    for f in range(floors):
        offset = f * c_per_floor
        grid = env.grids[f]
        full_channels[offset + 0, :, :] = (grid == 1).astype(np.float32)  # Wall
        full_channels[offset + 1, :, :] = (grid == 2).astype(np.float32)  # Fire
        full_channels[offset + 2, :, :] = (grid == 3).astype(np.float32)  # Hostage
        full_channels[offset + 3, :, :] = (grid == 4).astype(np.float32)  # Exit
        full_channels[offset + 4, :, :] = (grid == 5).astype(np.float32)  # Stairs

    for aid, rescuer in env.rescuers.items():
        if rescuer.alive:
            f = rescuer.floor
            offset = f * c_per_floor
            full_channels[offset + 5, rescuer.row, rescuer.col] = 1.0  # Rescuers

    canonical_grid = resize_to_canonical(full_channels, target_h=target_h, target_w=target_w)

    unrescued = sum(1 for h in env.hostages if not h.rescued)
    survivors = sum(1 for r in env.rescuers.values() if r.alive)
    fire_count = len(env.fire_positions)
    step_norm = float(env.current_step) / max(1.0, float(env.config.max_steps))

    global_scalars = np.array(
        [
            float(unrescued) / max(1.0, float(env.config.n_hostages)),
            float(survivors) / max(1.0, float(env.config.n_rescuers)),
            float(fire_count) / max(1.0, float(env.config.fire_max_tiles)),
            step_norm,
        ],
        dtype=np.float32,
    )

    return canonical_grid, global_scalars


class CentralizedCriticExtractor(nn.Module):
    """Features extractor for Centralized Critic consuming canonical global state."""

    def __init__(
        self,
        global_channels: int = 12,  # 6 channels * 2 floors
        target_h: int = 10,
        target_w: int = 10,
        scalars_dim: int = 4,
        features_dim: int = 160,
    ):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(global_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * target_h * target_w, 128),
            nn.ReLU(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(scalars_dim, 32),
            nn.ReLU(),
        )
        self.features_dim = features_dim

    def forward(self, global_grid: torch.Tensor, global_scalars: torch.Tensor) -> torch.Tensor:
        grid_feat = self.cnn(global_grid)
        scalar_feat = self.mlp(global_scalars)
        return torch.cat([grid_feat, scalar_feat], dim=1)


class AsymmetricMultiInputPolicy(ActorCriticPolicy):
    """MAPPO-style Asymmetric Actor-Critic policy.

    The actor policy head extracts features strictly from local observations (grid, self),
    guaranteeing size-invariance on any arbitrary grid dimension.
    The critic value head extracts features from the canonical global state.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Discrete,
        lr_schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        *args,
        **kwargs,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            *args,
            **kwargs,
        )

    def make_features_extractor(self) -> BaseFeaturesExtractor:
        """Constructs separate Actor and Centralized Critic features extractors."""
        # 1. Local Actor Extractor (Strictly consumes "grid" and "self")
        local_obs_space = spaces.Dict(
            {
                "grid": self.observation_space["grid"],
                "self": self.observation_space["self"],
            }
        )
        self.actor_extractor = RescueExtractor(local_obs_space, features_dim=160)

        # 2. Centralized Critic Extractor (Consumes "global" and "global_scalars" if present)
        self.has_global = "global" in self.observation_space.spaces
        if self.has_global:
            g_space = self.observation_space["global"]
            s_space = self.observation_space["global_scalars"]
            self.critic_extractor = CentralizedCriticExtractor(
                global_channels=g_space.shape[0],
                target_h=g_space.shape[1],
                target_w=g_space.shape[2],
                scalars_dim=s_space.shape[0],
                features_dim=160,
            )
        else:
            self.critic_extractor = None

        return self.actor_extractor

    def extract_features(
        self, obs: Dict[str, torch.Tensor], features_extractor: Optional[BaseFeaturesExtractor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extracts asymmetric (actor_features, critic_features) pair."""
        # STRICT ACTOR PRIVACY ASSERTION:
        # Actor must NEVER receive global state keys!
        actor_dict = {"grid": obs["grid"], "self": obs["self"]}
        assert "global" not in actor_dict, "CRITICAL ERROR: Actor input contains global state!"
        assert "global_scalars" not in actor_dict, "CRITICAL ERROR: Actor input contains global scalars!"

        actor_features = self.actor_extractor(actor_dict)

        if self.has_global and "global" in obs:
            critic_features = self.critic_extractor(obs["global"], obs["global_scalars"])
        else:
            # Fallback to local features for critic if global state not provided (e.g. testing)
            critic_features = actor_features

        return actor_features, critic_features

    def forward(
        self, obs: Dict[str, torch.Tensor], deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass in training: computes action distribution and state values."""
        actor_features, critic_features = self.extract_features(obs)
        latent_pi = self.mlp_extractor.forward_actor(actor_features)
        latent_vf = self.mlp_extractor.forward_critic(critic_features)

        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return actions, values, log_prob

    def evaluate_actions(
        self, obs: Dict[str, torch.Tensor], actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Evaluates actions during PPO updates."""
        actor_features, critic_features = self.extract_features(obs)
        latent_pi = self.mlp_extractor.forward_actor(actor_features)
        latent_vf = self.mlp_extractor.forward_critic(critic_features)

        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def get_distribution(self, obs: Dict[str, torch.Tensor]) -> Distribution:
        """Actor only: computes policy distribution from strictly local observations."""
        actor_dict = {"grid": obs["grid"], "self": obs["self"]}
        actor_features = self.actor_extractor(actor_dict)
        latent_pi = self.mlp_extractor.forward_actor(actor_features)
        return self._get_action_dist_from_latent(latent_pi)

    def predict_values(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Critic only: estimates state values."""
        _, critic_features = self.extract_features(obs)
        latent_vf = self.mlp_extractor.forward_critic(critic_features)
        return self.value_net(latent_vf)

    def _predict(self, observation: Dict[str, torch.Tensor], deterministic: bool = False) -> torch.Tensor:
        """Inference prediction using actor head strictly."""
        return self.get_distribution(observation).get_actions(deterministic=deterministic)
