"""Evaluation and metrics logging callback for RADON PPO training with train/test generalization protocol."""

import csv
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from env.config import EnvConfig
from env.rescue_env import RadonRescueEnv
from train.eval_protocol import TRAIN_EVAL_SEEDS, TEST_EVAL_SEEDS


class MetricsCallback(BaseCallback):
    """Custom callback recording headline rescue metrics and generalization gap to CSV.

    Metrics recorded:
      - timestep: Total environment timesteps so far.
      - train_success: Fraction of episodes where ALL hostages are evacuated on train seeds.
      - test_success: Fraction of episodes where ALL hostages are evacuated on held-out test seeds.
      - train_rescued: Average hostages rescued on train seeds.
      - test_rescued: Average hostages rescued on held-out test seeds.
      - gap_success: Generalization gap (train_success - test_success).
      - gap_rescued: Hostage rescue gap (train_rescued - test_rescued).
      - train_reward: Average return on train seeds.
      - test_reward: Average return on held-out test seeds.
      - mean_episode_length: Average steps per episode.
      - mean_rescuers_surviving: Average living rescuers remaining at termination.
    """

    def __init__(
        self,
        eval_freq: int = 10000,
        eval_episodes: int = 10,
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
        filename = f"{run_name}.csv"
        self.csv_path = self.log_dir / filename
        self.eval_deterministic = eval_deterministic
        self.last_eval_step = 0

        # Select evaluation seeds
        self.train_seeds = TRAIN_EVAL_SEEDS[:eval_episodes]
        self.test_seeds = TEST_EVAL_SEEDS[:eval_episodes]

        # Initialize CSV header if file doesn't exist
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "timestep",
                        "train_success",
                        "test_success",
                        "train_rescued",
                        "test_rescued",
                        "gap_success",
                        "gap_rescued",
                        "train_reward",
                        "test_reward",
                        "mean_episode_length",
                        "mean_rescuers_surviving",
                        "survival_rate",
                        "fire_deaths",
                    ]
                )

    def _evaluate_seeds(self, active_model, seeds: List[int]) -> Dict[str, float]:
        """Runs evaluation rollout episodes on a specific list of seeds."""
        ep_rewards = []
        ep_rescued = []
        ep_successes = []
        ep_lengths = []
        ep_surviving = []
        ep_fire_deaths = []

        eval_env = RadonRescueEnv(config=self.config)

        for seed in seeds:
            obs, infos = eval_env.reset(seed=seed)
            ep_return = 0.0

            while eval_env.agents:
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
            fire_deaths_count = eval_env.fire_deaths

            ep_rewards.append(ep_return)
            ep_rescued.append(rescued_count)
            ep_successes.append(1.0 if all_rescued else 0.0)
            ep_lengths.append(eval_env.current_step)
            ep_surviving.append(surviving_count)
            ep_fire_deaths.append(fire_deaths_count)

        eval_env.close()

        return {
            "reward": float(np.mean(ep_rewards)),
            "rescued": float(np.mean(ep_rescued)),
            "success": float(np.mean(ep_successes)),
            "length": float(np.mean(ep_lengths)),
            "surviving": float(np.mean(ep_surviving)),
            "survival_rate": float(np.mean(ep_surviving)) / float(max(1, self.config.n_rescuers)),
            "fire_deaths": float(np.mean(ep_fire_deaths)),
        }

    def _evaluate_policy(self, model=None, num_episodes: Optional[int] = None) -> Dict[str, float]:
        """Evaluates policy across both TRAIN seeds and held-out TEST seeds."""
        active_model = model if model is not None else self.model
        n_eps = num_episodes if num_episodes is not None else self.eval_episodes
        train_seeds = TRAIN_EVAL_SEEDS[:n_eps]
        test_seeds = TEST_EVAL_SEEDS[:n_eps]

        train_res = self._evaluate_seeds(active_model, train_seeds)
        test_res = self._evaluate_seeds(active_model, test_seeds)

        gap_succ = train_res["success"] - test_res["success"]
        gap_resc = train_res["rescued"] - test_res["rescued"]
        mean_surv = (train_res["surviving"] + test_res["surviving"]) / 2.0

        return {
            "train_success": train_res["success"],
            "test_success": test_res["success"],
            "train_rescued": train_res["rescued"],
            "test_rescued": test_res["rescued"],
            "gap_success": gap_succ,
            "gap_rescued": gap_resc,
            "train_reward": train_res["reward"],
            "test_reward": test_res["reward"],
            "mean_episode_length": (train_res["length"] + test_res["length"]) / 2.0,
            "mean_rescuers_surviving": mean_surv,
            "survival_rate": mean_surv / float(max(1, self.config.n_rescuers)),
            "fire_deaths": (train_res["fire_deaths"] + test_res["fire_deaths"]) / 2.0,
        }

    def _log_metrics(self, timestep: int, metrics: Dict[str, float]) -> None:
        """Appends metrics to CSV and prints clean log."""
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    timestep,
                    round(metrics["train_success"], 4),
                    round(metrics["test_success"], 4),
                    round(metrics["train_rescued"], 4),
                    round(metrics["test_rescued"], 4),
                    round(metrics["gap_success"], 4),
                    round(metrics["gap_rescued"], 4),
                    round(metrics["train_reward"], 4),
                    round(metrics["test_reward"], 4),
                    round(metrics["mean_episode_length"], 2),
                    round(metrics["mean_rescuers_surviving"], 2),
                    round(metrics["survival_rate"], 4),
                    round(metrics["fire_deaths"], 2),
                ]
            )

        if self.verbose > 0:
            print(
                f"[Eval @ {timestep:6d} steps] "
                f"TRAIN: Succ {metrics['train_success']*100:5.1f}% ({metrics['train_rescued']:4.2f}/{self.config.n_hostages}) | "
                f"TEST (Held-out): Succ {metrics['test_success']*100:5.1f}% ({metrics['test_rescued']:4.2f}/{self.config.n_hostages}) | "
                f"Gap: {metrics['gap_success']*100:+5.1f}% | "
                f"Surv: {metrics['mean_rescuers_surviving']:3.1f}/{self.config.n_rescuers} ({metrics['survival_rate']*100:.0f}%) | "
                f"FireDeaths: {metrics['fire_deaths']:.1f}"
            )

    def _on_training_start(self) -> None:
        """Evaluates baseline before learning begins."""
        metrics = self._evaluate_policy()
        self._log_metrics(0, metrics)

    def _on_step(self) -> bool:
        """Triggered at each environment step."""
        if self.num_timesteps - self.last_eval_step >= self.eval_freq:
            self.last_eval_step = self.num_timesteps
            metrics = self._evaluate_policy()
            self._log_metrics(self.num_timesteps, metrics)

        return True
