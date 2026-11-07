"""Evaluation and metrics logging callback for RADON PPO training."""

import csv
from pathlib import Path
from typing import Optional
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from env.config import EnvConfig
from env.rescue_env import RadonRescueEnv


class MetricsCallback(BaseCallback):
    """Custom callback recording headline rescue metrics to CSV every eval_freq steps.

    Metrics recorded:
      - timestep: Total environment timesteps so far.
      - mean_reward: Average cumulative episode return across all agents.
      - mean_hostages_rescued: Average number of hostages successfully evacuated.
      - success_rate: Fraction of episodes where ALL hostages are evacuated.
      - mean_episode_length: Average steps per episode.
      - mean_rescuers_surviving: Average living rescuers remaining at termination.
    """

    def __init__(
        self,
        eval_freq: int = 10000,
        eval_episodes: int = 5,
        config: Optional[EnvConfig] = None,
        log_dir: str = "metrics",
        run_name: str = "default",
        eval_deterministic: bool = False,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.config = config if config is not None else EnvConfig.training_day2()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.log_dir / f"day2_{run_name}.csv"
        self.eval_deterministic = eval_deterministic
        self.last_eval_step = 0

        # Initialize CSV header if file doesn't exist
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "timestep",
                        "mean_reward",
                        "mean_hostages_rescued",
                        "success_rate",
                        "mean_episode_length",
                        "mean_rescuers_surviving",
                    ]
                )

    def _evaluate_policy(self, model=None, num_episodes: Optional[int] = None) -> dict[str, float]:
        """Runs evaluation rollout episodes on deterministic seeds."""
        active_model = model if model is not None else self.model
        n_eps = num_episodes if num_episodes is not None else self.eval_episodes
        ep_rewards = []
        ep_rescued = []
        ep_successes = []
        ep_lengths = []
        ep_surviving = []

        eval_env = RadonRescueEnv(config=self.config)

        for ep_idx in range(n_eps):
            seed = self.config.seed if self.config.seed is not None else (10000 + ep_idx)
            obs, infos = eval_env.reset(seed=seed)
            ep_return = 0.0

            while eval_env.agents:
                # Predict action for each active agent using shared PPO policy
                actions = {}
                for agent_id in eval_env.agents:
                    agent_obs = {
                        k: np.expand_dims(v, axis=0) for k, v in obs[agent_id].items()
                    }
                    action, _ = active_model.predict(agent_obs, deterministic=self.eval_deterministic)
                    actions[agent_id] = int(action[0])

                obs, rewards, terminations, truncations, infos = eval_env.step(actions)
                ep_return += sum(rewards.values())

            rescued_count = sum(1 for h in eval_env.hostages if h.rescued)
            all_rescued = (rescued_count == self.config.n_hostages)
            surviving_count = sum(1 for r in eval_env.rescuers.values() if r.alive)

            ep_rewards.append(ep_return)
            ep_rescued.append(rescued_count)
            ep_successes.append(1.0 if all_rescued else 0.0)
            ep_lengths.append(eval_env.current_step)
            ep_surviving.append(surviving_count)

        eval_env.close()

        return {
            "mean_reward": float(np.mean(ep_rewards)),
            "mean_hostages_rescued": float(np.mean(ep_rescued)),
            "success_rate": float(np.mean(ep_successes)),
            "mean_episode_length": float(np.mean(ep_lengths)),
            "mean_rescuers_surviving": float(np.mean(ep_surviving)),
        }

    def _on_training_start(self) -> None:
        """Evaluates baseline before learning begins."""
        metrics = self._evaluate_policy()
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    0,
                    round(metrics["mean_reward"], 4),
                    round(metrics["mean_hostages_rescued"], 4),
                    round(metrics["success_rate"], 4),
                    round(metrics["mean_episode_length"], 2),
                    round(metrics["mean_rescuers_surviving"], 2),
                ]
            )
        if self.verbose > 0:
            print(
                f"[Baseline @      0 steps] "
                f"Success: {metrics['success_rate']*100:5.1f}% | "
                f"Hostages: {metrics['mean_hostages_rescued']:4.2f}/{self.config.n_hostages} | "
                f"Return: {metrics['mean_reward']:+7.2f} | "
                f"Surviving: {metrics['mean_rescuers_surviving']:3.1f}/{self.config.n_rescuers}"
            )

    def _on_step(self) -> bool:
        """Triggered at each environment step."""
        if self.num_timesteps - self.last_eval_step >= self.eval_freq:
            self.last_eval_step = self.num_timesteps
            metrics = self._evaluate_policy()

            # Append to CSV
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        self.num_timesteps,
                        round(metrics["mean_reward"], 4),
                        round(metrics["mean_hostages_rescued"], 4),
                        round(metrics["success_rate"], 4),
                        round(metrics["mean_episode_length"], 2),
                        round(metrics["mean_rescuers_surviving"], 2),
                    ]
                )

            if self.verbose > 0:
                print(
                    f"[Eval @ {self.num_timesteps:6d} steps] "
                    f"Success: {metrics['success_rate']*100:5.1f}% | "
                    f"Hostages: {metrics['mean_hostages_rescued']:4.2f}/{self.config.n_hostages} | "
                    f"Return: {metrics['mean_reward']:+7.2f} | "
                    f"Surviving: {metrics['mean_rescuers_surviving']:3.1f}/{self.config.n_rescuers}"
                )

        return True
