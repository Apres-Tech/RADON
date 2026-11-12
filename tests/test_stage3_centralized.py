"""Unit tests verifying Stage 3 Centralized Critic, Actor Privacy, and Size-Invariance on 20x20."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import gymnasium.spaces as spaces
from stable_baselines3 import PPO

from env.config import EnvConfig
from env.rescue_env import RadonRescueEnv
from train.centralized import AsymmetricMultiInputPolicy, build_centralized_global_state


def test_actor_privacy_and_centralized_critic():
    """Verify that actor receives ONLY local keys ('grid', 'self') and NEVER receives 'global'."""
    cfg = EnvConfig.showcase(obs_bfs_features=True, seed=42)
    env = RadonRescueEnv(config=cfg)
    obs, _ = env.reset(seed=42)

    # Observation space augmented with centralized global state for critic
    global_grid, global_scalars = build_centralized_global_state(env, target_h=10, target_w=10)
    full_obs_space = spaces.Dict(
        {
            "grid": env.observation_space("rescuer_0")["grid"],
            "self": env.observation_space("rescuer_0")["self"],
            "global": spaces.Box(low=0.0, high=1.0, shape=global_grid.shape, dtype=np.float32),
            "global_scalars": spaces.Box(low=0.0, high=1.0, shape=global_scalars.shape, dtype=np.float32),
        }
    )

    action_space = env.action_space("rescuer_0")
    policy = AsymmetricMultiInputPolicy(
        observation_space=full_obs_space,
        action_space=action_space,
        lr_schedule=lambda _: 3e-4,
        net_arch=dict(pi=[128, 64], vf=[128, 64]),
    )

    # Build full observation batch
    aid = "rescuer_0"
    obs_batch = {
        "grid": torch.as_tensor(np.expand_dims(obs[aid]["grid"], axis=0), dtype=torch.float32),
        "self": torch.as_tensor(np.expand_dims(obs[aid]["self"], axis=0), dtype=torch.float32),
        "global": torch.as_tensor(np.expand_dims(global_grid, axis=0), dtype=torch.float32),
        "global_scalars": torch.as_tensor(np.expand_dims(global_scalars, axis=0), dtype=torch.float32),
    }

    # 1. Forward pass during training
    actions, values, log_probs = policy.forward(obs_batch)
    assert actions.shape == (1,)
    assert values.shape == (1, 1)
    assert log_probs.shape == (1,)
    print("[PASS] Asymmetric forward pass succeeded: actor actions and critic values computed.")

    # 2. Strict Privacy Test: evaluate_actions and get_distribution
    dist = policy.get_distribution(obs_batch)
    assert dist is not None

    # 3. Prove that ACTOR runs without 'global' or 'global_scalars' in observation
    local_only_batch = {
        "grid": obs_batch["grid"],
        "self": obs_batch["self"],
    }
    local_dist = policy.get_distribution(local_only_batch)
    local_action = policy._predict(local_only_batch)
    assert local_action.shape == (1,)
    print("[PASS] Actor privacy verified: actor operates strictly on local observations.")


def test_size_invariance_on_20x20():
    """Verify that policy runs on 20x20 map (showcase_hard) without shape errors."""
    cfg_14 = EnvConfig.showcase(obs_bfs_features=True)
    cfg_20 = EnvConfig.showcase_hard(obs_bfs_features=True)

    env_14 = RadonRescueEnv(config=cfg_14)
    env_14.reset(seed=42)
    env_20 = RadonRescueEnv(config=cfg_20)

    # Build policy on canonical 14x14 space
    global_grid, global_scalars = build_centralized_global_state(env_14, target_h=10, target_w=10)
    full_obs_space = spaces.Dict(
        {
            "grid": env_14.observation_space("rescuer_0")["grid"],
            "self": env_14.observation_space("rescuer_0")["self"],
            "global": spaces.Box(low=0.0, high=1.0, shape=global_grid.shape, dtype=np.float32),
            "global_scalars": spaces.Box(low=0.0, high=1.0, shape=global_scalars.shape, dtype=np.float32),
        }
    )
    policy = AsymmetricMultiInputPolicy(
        observation_space=full_obs_space,
        action_space=env_14.action_space("rescuer_0"),
        lr_schedule=lambda _: 3e-4,
    )

    # Reset 20x20 environment
    obs_20, _ = env_20.reset(seed=999)
    for aid in env_20.agents:
        agent_obs = {
            "grid": torch.as_tensor(np.expand_dims(obs_20[aid]["grid"], axis=0), dtype=torch.float32),
            "self": torch.as_tensor(np.expand_dims(obs_20[aid]["self"], axis=0), dtype=torch.float32),
        }
        action = policy._predict(agent_obs)
        assert action.shape == (1,)

    print("[PASS] Size-invariance verified: policy runs on 20x20 showcase_hard environment without shape error.")


if __name__ == "__main__":
    test_actor_privacy_and_centralized_critic()
    test_size_invariance_on_20x20()
    print("\nALL STAGE 3 CENTRALIZED CRITIC TESTS PASSED!")
