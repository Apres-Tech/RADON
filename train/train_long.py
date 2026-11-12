"""Resumable multi-day trainer for RADON rescue agents with held-out evaluation tracking.

Supports:
- Checkpoints saved to agents/ckpt_<arch>_<steps>.zip
- Resuming seamlessly with --resume [path|latest] preserving step count
- Held-out seed evaluation (50000..50029) and appending to metrics/training_<arch>.csv
- Tracking and saving agents/best_<arch>.zip
- Throughput and wall-clock logging
- Architectures: mlp (baseline), bfs_mlp (Stage 1), recurrent (Stage 2), centralized (Stage 3)
"""

import argparse
import csv
import glob
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from env.config import EnvConfig
from env.fire_controller import CurriculumFireController
from train.diagnose import run_diagnostics, print_diagnostic_table
from train.extractor import RescueExtractor
from train.make_env import make_rescue_vec_env


AGENTS_DIR = Path("agents")
METRICS_DIR = Path("metrics")


class LongRunCallback(BaseCallback):
    """Tracks checkpoints, held-out evaluation metrics, wall-clock time, and best model."""

    def __init__(
        self,
        arch: str,
        checkpoint_freq: int = 250_000,
        eval_freq: int = 250_000,
        eval_episodes: int = 30,
        base_seed: int = 50000,
        preset: str = "showcase",
        start_step: int = 0,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.arch = arch
        self.checkpoint_freq = checkpoint_freq
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.base_seed = base_seed
        self.preset = preset
        self.start_step = start_step

        self.last_checkpoint_step = start_step
        self.last_eval_step = start_step
        self.best_test_success = -1.0
        self.best_test_rescued = -1.0

        self.start_time = time.time()
        self.last_time = self.start_time
        self.last_step_count = start_step

        AGENTS_DIR.mkdir(parents=True, exist_ok=True)
        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        self.csv_path = METRICS_DIR / f"training_{arch}.csv"

        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "timestep",
                        "train_success",
                        "test_success",
                        "test_rescued",
                        "survival",
                        "fire_deaths",
                        "oscillation_rate",
                        "wall_time_sec",
                        "steps_per_sec",
                    ]
                )

    def _on_step(self) -> bool:
        current_step = self.start_step + self.num_timesteps

        # 1. Checkpoint save
        if current_step - self.last_checkpoint_step >= self.checkpoint_freq:
            self.last_checkpoint_step = current_step
            ckpt_path = AGENTS_DIR / f"ckpt_{self.arch}_{current_step}.zip"
            print(f"\n[Checkpoint] Saving periodic checkpoint to {ckpt_path}...")
            self.model.save(str(ckpt_path))

        # 2. Held-out Evaluation
        if current_step - self.last_eval_step >= self.eval_freq:
            self.last_eval_step = current_step
            now = time.time()
            elapsed = now - self.start_time
            step_delta = current_step - self.last_step_count
            time_delta = max(1e-5, now - self.last_time)
            sps = step_delta / time_delta
            self.last_time = now
            self.last_step_count = current_step

            print(f"\n[Evaluation @ Step {current_step:,}] Evaluating on held-out seeds {self.base_seed}..{self.base_seed + self.eval_episodes - 1}...")
            diag = run_diagnostics(
                model_obj=self.model,
                num_episodes=self.eval_episodes,
                base_seed=self.base_seed,
                preset=self.preset,
                deterministic=False,
            )
            print_diagnostic_table(f"Step {current_step:,} ({self.arch}) Held-out Evaluation", diag)

            # Append to CSV
            test_success = diag["full_win_rate"]
            test_rescued = diag["mean_rescued"]
            survival = diag["mean_surviving"]
            fire_deaths = diag["mean_fire_deaths"]
            osc_rate = diag["oscillation_rate"]
            # Train success placeholder (or diag if evaluated on train seeds)
            train_success = test_success

            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        current_step,
                        f"{train_success:.1f}",
                        f"{test_success:.1f}",
                        f"{test_rescued:.2f}",
                        f"{survival:.2f}",
                        f"{fire_deaths:.2f}",
                        f"{osc_rate:.1f}",
                        f"{elapsed:.1f}",
                        f"{sps:.1f}",
                    ]
                )

            # Check if new best
            if (test_success > self.best_test_success) or (
                test_success == self.best_test_success and test_rescued > self.best_test_rescued
            ):
                self.best_test_success = test_success
                self.best_test_rescued = test_rescued
                best_path = AGENTS_DIR / f"best_{self.arch}.zip"
                print(f"[BEST MODEL] New best held-out model saved: {best_path} (Success: {test_success:.1f}%, Rescued: {test_rescued:.2f})")
                self.model.save(str(best_path))

        return True


def find_latest_checkpoint(arch: str) -> Optional[Tuple[Path, int]]:
    """Finds the checkpoint with the highest step count for the given architecture."""
    pattern = str(AGENTS_DIR / f"ckpt_{arch}_*.zip")
    matches = glob.glob(pattern)
    if not matches:
        # Check if best_<arch>.zip exists
        best = AGENTS_DIR / f"best_{arch}.zip"
        if best.exists():
            return best, 0
        return None

    best_match = None
    max_steps = -1
    for m in matches:
        match = re.search(r"ckpt_" + re.escape(arch) + r"_(\d+)\.zip", m)
        if match:
            steps = int(match.group(1))
            if steps > max_steps:
                max_steps = steps
                best_match = Path(m)

    if best_match is not None:
        return best_match, max_steps
    return None


def get_model_cls_and_policy(arch: str):
    """Returns (model_class, policy_type, policy_kwargs) for the architecture."""
    if arch == "recurrent":
        from sb3_contrib import RecurrentPPO
        policy_kwargs = dict(
            features_extractor_class=RescueExtractor,
            features_extractor_kwargs=dict(features_dim=160),
            net_arch=dict(pi=[128], vf=[128]),
            lstm_hidden_size=128,
            n_lstm_layers=1,
            shared_lstm=False,
            enable_critic_lstm=True,
        )
        return RecurrentPPO, "MultiInputLstmPolicy", policy_kwargs
    elif arch == "centralized":
        # Stage 3 MAPPO asymmetric actor-critic policy (defined in train/centralized.py)
        from train.centralized import AsymmetricMultiInputPolicy
        policy_kwargs = dict(
            features_extractor_class=RescueExtractor,
            features_extractor_kwargs=dict(features_dim=160),
            net_arch=dict(pi=[128, 64], vf=[128, 64]),
        )
        return PPO, AsymmetricMultiInputPolicy, policy_kwargs
    else:
        # mlp or bfs_mlp
        policy_kwargs = dict(
            features_extractor_class=RescueExtractor,
            features_extractor_kwargs=dict(features_dim=160),
            net_arch=dict(pi=[128, 64], vf=[128, 64]),
        )
        return PPO, "MultiInputPolicy", policy_kwargs


def train_long(
    arch: str = "bfs_mlp",
    total_timesteps: int = 2_000_000,
    checkpoint_freq: int = 250_000,
    eval_freq: int = 250_000,
    eval_episodes: int = 30,
    resume: Optional[str] = None,
    num_vec_envs: int = 4,
    preset: str = "showcase",
    lr: float = 3e-4,
    batch_size: int = 64,
    n_steps: int = 1024,
    ent_coef: float = 0.02,
    device: Optional[str] = None,
):
    """Executes long resumable training loop with held-out metrics logging."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 76)
    print(f"  RADON Long-Run Resumable Trainer — Arch: {arch}")
    print(f"  Target Timesteps: {total_timesteps:,} | Parallel Envs: {num_vec_envs} | Device: {device}")
    print("=" * 76)

    # Resolve environment config
    if preset == "showcase":
        cfg = EnvConfig.showcase()
    elif preset == "showcase_hard":
        cfg = EnvConfig.showcase_hard()
    else:
        cfg = EnvConfig.training_day4_base()

    def _fc_factory():
        return CurriculumFireController()

    vec_env = make_rescue_vec_env(
        config=cfg,
        num_vec_envs=num_vec_envs,
        fire_controller_factory=_fc_factory,
    )

    model_cls, policy_type, policy_kwargs = get_model_cls_and_policy(arch)

    # Resume handling
    resume_path = None
    start_step = 0
    if resume:
        if resume.lower() == "latest":
            latest = find_latest_checkpoint(arch)
            if latest:
                resume_path, start_step = latest
                print(f"[Resume] Found latest checkpoint: {resume_path} at step {start_step:,}")
            else:
                print(f"[Resume] No existing checkpoint found for arch '{arch}'. Starting fresh.")
        else:
            p = Path(resume)
            if p.exists():
                resume_path = p
                match = re.search(r"(\d+)\.zip", p.name)
                start_step = int(match.group(1)) if match else 0
                print(f"[Resume] Loading checkpoint from {resume_path} at step {start_step:,}")
            else:
                raise FileNotFoundError(f"Checkpoint not found: {resume}")

    if resume_path is not None:
        model = model_cls.load(
            str(resume_path),
            env=vec_env,
            device=device,
            tensorboard_log=None,
        )
        print(f"[Model] Resumed {arch} model from {resume_path}")
    else:
        model = model_cls(
            policy_type,
            vec_env,
            learning_rate=lr,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=ent_coef,
            policy_kwargs=policy_kwargs,
            verbose=1,
            device=device,
        )
        print(f"[Model] Initialized fresh {arch} model ({policy_type})")

    remaining_steps = max(0, total_timesteps - start_step)
    print(f"[Planner] Training for {remaining_steps:,} more steps (current: {start_step:,} / target: {total_timesteps:,})...")

    callback = LongRunCallback(
        arch=arch,
        checkpoint_freq=checkpoint_freq,
        eval_freq=eval_freq,
        eval_episodes=eval_episodes,
        base_seed=50000,
        preset=preset,
        start_step=start_step,
    )

    if remaining_steps > 0:
        model.learn(
            total_timesteps=remaining_steps,
            callback=callback,
            reset_num_timesteps=False,
        )

    # Final save and eval
    final_step = start_step + remaining_steps
    final_save_path = AGENTS_DIR / f"ckpt_{arch}_{final_step}.zip"
    model.save(str(final_save_path))
    print(f"\n[Completed] Saved final checkpoint: {final_save_path}")

    # Run final diagnostics
    print(f"\n[Final Benchmark] Running failure diagnostics on held-out seeds...")
    diag = run_diagnostics(
        model_obj=model,
        num_episodes=eval_episodes,
        base_seed=50000,
        preset=preset,
        deterministic=False,
    )
    print_diagnostic_table(f"Final {arch} @ {final_step:,} Steps Held-out Evaluation", diag)

    vec_env.close()
    return model, diag


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RADON Resumable Long Trainer")
    parser.add_argument(
        "--arch",
        type=str,
        default="bfs_mlp",
        choices=["mlp", "bfs_mlp", "recurrent", "centralized"],
        help="Architecture variant",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=2_000_000,
        help="Total target timesteps",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=250_000,
        help="Checkpoint save frequency",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=250_000,
        help="Held-out eval frequency",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=30,
        help="Number of held-out eval episodes",
    )
    parser.add_argument(
        "--num-vec-envs",
        type=int,
        default=4,
        help="Number of parallel environments",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="showcase",
        choices=["showcase", "showcase_hard", "training_day4_base"],
        help="Config preset",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint or 'latest'",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=1024,
        help="Steps per environment per rollout",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.02,
        help="Entropy coefficient",
    )
    args = parser.parse_args()

    train_long(
        arch=args.arch,
        total_timesteps=args.timesteps,
        checkpoint_freq=args.checkpoint_freq,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        resume=args.resume,
        num_vec_envs=args.num_vec_envs,
        preset=args.preset,
        lr=args.lr,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        ent_coef=args.ent_coef,
    )
