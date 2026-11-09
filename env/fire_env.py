"""Gymnasium Environment for training the Fire Adversary in RADON self-play."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from env.config import EnvConfig
from env.fire_controller import build_fire_observation, get_valid_fire_frontier, select_frontier_ignition
from env.rescue_env import RadonRescueEnv
from env.tiles import Tile


class FireAdversaryEnv(gym.Env):
    """Single-agent Gymnasium environment where the agent IS the Fire adversary.

    The fire policy plays against a frozen rescuer team policy, learning to place
    tactical ignitions along the fire frontier to block corridors, trap rescuers,
    and prevent hostage evacuation.
    """

    metadata = {"render_modes": ["ansi", "human"]}

    def __init__(
        self,
        config: Optional[EnvConfig] = None,
        rescuer_policy_path: Optional[str] = "agents/base_frozen.zip",
        rescuer_model=None,
        deterministic_rescuers: bool = False,
    ):
        super().__init__()
        self.config = config if config is not None else EnvConfig.training_day4_base()
        self.deterministic_rescuers = deterministic_rescuers

        # Discrete action space: 8 tactical frontier choices
        self.action_space = spaces.Discrete(8)

        # Multi-input observation space: global grid + state scalars
        floors = self.config.floors
        h, w = self.config.grid_h, self.config.grid_w
        self.observation_space = spaces.Dict(
            {
                "grid": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(6 * floors, h, w),
                    dtype=np.float32,
                ),
                "scalars": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(6,),
                    dtype=np.float32,
                ),
            }
        )

        # Rescuer environment (fire_controller=None because this environment drives fire directly)
        self.rescue_env = RadonRescueEnv(config=self.config, fire_controller=None)

        # Load frozen rescuer policy
        self.rescuer_model = rescuer_model
        if self.rescuer_model is None and rescuer_policy_path is not None:
            p = Path(rescuer_policy_path)
            if p.exists():
                from stable_baselines3 import PPO
                self.rescuer_model = PPO.load(str(p))

        self.last_rescuer_obs: Dict[str, Any] = {}
        self.current_step: int = 0
        self.frontier: List[Tuple[int, int, int]] = []
        self.prev_rescued_count: int = 0
        self.prev_fire_deaths: int = 0

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        self.last_rescuer_obs, infos = self.rescue_env.reset(seed=seed)
        self.current_step = 0
        self.prev_rescued_count = 0
        self.prev_fire_deaths = 0

        self.frontier = get_valid_fire_frontier(
            self.rescue_env.grids,
            self.rescue_env.fire_positions,
            self.config.grid_h,
            self.config.grid_w,
        )

        fire_obs = build_fire_observation(
            grids=self.rescue_env.grids,
            fire_positions=self.rescue_env.fire_positions,
            rescuers=self.rescue_env.rescuers,
            hostages=self.rescue_env.hostages,
            current_step=self.current_step,
            config=self.config,
            frontier=self.frontier,
        )

        info = {
            "unrescued_hostages": sum(1 for h in self.rescue_env.hostages if not h.rescued),
            "living_rescuers": sum(1 for r in self.rescue_env.rescuers.values() if r.alive),
            "fire_tiles": len(self.rescue_env.fire_positions),
        }
        return fire_obs, info

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        self.current_step += 1

        # --- 1. Adversary Action: Tactical Frontier Ignition ---
        if (
            self.config.fire_enabled
            and self.current_step >= self.config.fire_spread_start_step
            and len(self.rescue_env.fire_positions) < self.config.fire_max_tiles
        ):
            ignitions = select_frontier_ignition(
                action=int(action),
                frontier=self.frontier,
                rescuers=self.rescue_env.rescuers,
                hostages=self.rescue_env.hostages,
                config=self.config,
                budget=1,
            )
            ignited_any = False
            for f, r, c in ignitions:
                if self.rescue_env.grids[f, r, c] == Tile.EMPTY and len(self.rescue_env.fire_positions) < self.config.fire_max_tiles:
                    self.rescue_env.grids[f, r, c] = Tile.FIRE
                    self.rescue_env.fire_positions.append((f, r, c))
                    ignited_any = True

            if ignited_any:
                self.rescue_env._recompute_shaping_maps(update_exits=True)

        # --- 2. Step Rescuers Using Frozen Rescuer Policy ---
        rescuer_actions = {}
        for agent_id in self.rescue_env.agents:
            if self.rescuer_model is not None:
                agent_obs = {
                    k: np.expand_dims(v, axis=0)
                    for k, v in self.last_rescuer_obs[agent_id].items()
                }
                act, _ = self.rescuer_model.predict(agent_obs, deterministic=self.deterministic_rescuers)
                rescuer_actions[agent_id] = int(act[0])
            else:
                rescuer_actions[agent_id] = int(self.rescue_env.action_space(agent_id).sample())

        self.last_rescuer_obs, rewards, terminations, truncations, infos = self.rescue_env.step(rescuer_actions)

        # --- 3. Compute Zero-Sum Adversary Reward ---
        curr_rescued = sum(1 for h in self.rescue_env.hostages if h.rescued)
        new_deliveries = max(0, curr_rescued - self.prev_rescued_count)
        self.prev_rescued_count = curr_rescued

        curr_deaths = self.rescue_env.fire_deaths
        new_deaths = max(0, curr_deaths - self.prev_fire_deaths)
        self.prev_fire_deaths = curr_deaths

        unrescued = sum(1 for h in self.rescue_env.hostages if not h.rescued)
        living = sum(1 for r in self.rescue_env.rescuers.values() if r.alive)

        # Base step reward: fire benefits from keeping hostages trapped
        reward = 0.05 * float(unrescued)

        # Event rewards
        if new_deaths > 0:
            reward += 6.0 * new_deaths  # Strong reward for killing rescuer
        if new_deliveries > 0:
            reward -= 12.0 * new_deliveries  # Strong penalty when hostage is delivered

        all_rescued = (curr_rescued == self.config.n_hostages)
        all_dead = (living == 0)
        is_truncated = self.current_step >= self.config.max_steps
        is_terminated = all_rescued or all_dead

        if all_rescued:
            reward -= 20.0  # Rescuer team win -> fire heavy penalty
        elif all_dead:
            reward += 20.0  # Fire team wipe -> fire massive victory

        # Update frontier for next step
        self.frontier = get_valid_fire_frontier(
            self.rescue_env.grids,
            self.rescue_env.fire_positions,
            self.config.grid_h,
            self.config.grid_w,
        )

        fire_obs = build_fire_observation(
            grids=self.rescue_env.grids,
            fire_positions=self.rescue_env.fire_positions,
            rescuers=self.rescue_env.rescuers,
            hostages=self.rescue_env.hostages,
            current_step=self.current_step,
            config=self.config,
            frontier=self.frontier,
        )

        info = {
            "unrescued_hostages": unrescued,
            "living_rescuers": living,
            "fire_deaths": curr_deaths,
            "hostages_rescued": curr_rescued,
            "fire_tiles": len(self.rescue_env.fire_positions),
            "step": self.current_step,
        }

        return fire_obs, reward, is_terminated, is_truncated, info

