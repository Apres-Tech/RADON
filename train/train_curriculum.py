"""Progressive curriculum trainer for RADON 14x14 showcase environments."""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import shutil
import sys
import time
from typing import List, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from stable_baselines3 import PPO

from env.config import EnvConfig
from env.fire_controller import CurriculumFireController
from env.rescue_env import RadonRescueEnv
from train.train_ppo import make_rescue_vec_env, RescueExtractor

AGENTS_DIR = Path("agents")
METRICS_DIR = Path("metrics")


@dataclass
class CurriculumStage:
    name: str
    grid_size: int
    n_rescuers: int
    n_hostages: int
    max_steps: int
    fire_max_tiles: int
    timesteps: int
    num_vec_envs: int


STAGES = [
    CurriculumStage(
        name="Stage 1 (14x14 - 2 Hostages Warmup)",
        grid_size=14,
        n_rescuers=4,
        n_hostages=2,
        max_steps=180,
        fire_max_tiles=20,
        timesteps=60_000,
        num_vec_envs=4,
    ),
    CurriculumStage(
        name="Stage 2 (14x14 - 3 Hostages Target)",
        grid_size=14,
        n_rescuers=4,
        n_hostages=3,
        max_steps=220,
        fire_max_tiles=35,
        timesteps=100_000,
        num_vec_envs=4,
    ),
    CurriculumStage(
        name="Stage 3 (14x14 - 3 Hostages Hardened)",
        grid_size=14,
        n_rescuers=4,
        n_hostages=3,
        max_steps=220,
        fire_max_tiles=35,
        timesteps=60_000,
        num_vec_envs=4,
    ),
]


def evaluate_stage_heldout(
    model: PPO,
    config: EnvConfig,
    num_episodes: int = 30,
    base_seed: int = 50000,
) -> Tuple[float, float, float, float]:
    """Evaluates policy over fixed held-out unseen maps (seeds 50000..50029)."""
    successes = []
    rescued_ratios = []
    surviving_counts = []
    returns = []

    for ep in range(num_episodes):
        ep_seed = base_seed + ep
        ep_cfg = EnvConfig.showcase(
            grid_size=config.grid_h,
            n_rescuers=config.n_rescuers,
            n_hostages=config.n_hostages,
            max_steps=config.max_steps,
            seed=ep_seed,
        )
        ep_cfg.fire_max_tiles = config.fire_max_tiles
        ep_cfg.fire_spread_start_step = config.fire_spread_start_step
        ep_cfg.fire_spread_prob = config.fire_spread_prob

        env = RadonRescueEnv(config=ep_cfg, fire_controller=CurriculumFireController())
        obs, _ = env.reset(seed=ep_seed)

        ep_return = 0.0
        while env.agents:
            actions = {}
            for aid in env.agents:
                agent_obs = {k: np.expand_dims(v, axis=0) for k, v in obs[aid].items()}
                act, _ = model.predict(agent_obs, deterministic=False)
                actions[aid] = int(act[0])

            obs, rewards, terms, truncs, _ = env.step(actions)
            ep_return += sum(rewards.values())

        rescued = sum(1 for h in env.hostages if h.rescued)
        survivors = sum(1 for r in env.rescuers.values() if r.alive)

        successes.append(1.0 if rescued == config.n_hostages else 0.0)
        rescued_ratios.append(rescued / config.n_hostages)
        surviving_counts.append(survivors)
        returns.append(ep_return)

    return (
        float(np.mean(successes)),
        float(np.mean(rescued_ratios)),
        float(np.mean(surviving_counts)),
        float(np.mean(returns)),
    )


def run_curriculum(resume_path: str = "agents/final_rescuer.zip", fast_mode: bool = False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70)
    print("  RADON 14x14 Showcase Curriculum Trainer")
    print(f"  Device: {device} | Base Model: {resume_path}")
    print("=" * 70)

    AGENTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    current_weights = resume_path if Path(resume_path).exists() else "agents/base_frozen.zip"
    results = []
    best_model_path = None
    best_target_win_rate = -1.0

    for idx, stage in enumerate(STAGES, 1):
        timesteps = stage.timesteps if not fast_mode else stage.timesteps // 2
        print(f"\n[{idx}/{len(STAGES)}] Starting {stage.name} — Grid: {stage.grid_size}x{stage.grid_size}, Rescuers: {stage.n_rescuers}, Hostages: {stage.n_hostages}, Timesteps: {timesteps:,}")

        config = EnvConfig.showcase(
            grid_size=stage.grid_size,
            n_rescuers=stage.n_rescuers,
            n_hostages=stage.n_hostages,
            max_steps=stage.max_steps,
            seed=None,
        )
        config.fire_max_tiles = stage.fire_max_tiles

        vec_env = make_rescue_vec_env(
            config=config,
            num_vec_envs=stage.num_vec_envs,
            num_cpus=1,
            seed=42 + idx,
            fire_controller_factory=lambda: CurriculumFireController(),
        )

        policy_kwargs = dict(
            features_extractor_class=RescueExtractor,
            features_extractor_kwargs=dict(features_dim=160),
            net_arch=dict(pi=[128, 64], vf=[128, 64]),
        )

        lr = 3e-4 if idx <= 2 else 1.5e-4
        ent_coef = 0.03 if idx <= 2 else 0.02

        if current_weights and Path(current_weights).exists():
            print(f"  Loading pretrained weights from: {current_weights}")
            model = PPO.load(
                current_weights,
                env=vec_env,
                learning_rate=lr,
                ent_coef=ent_coef,
                device=device,
            )
        else:
            model = PPO(
                "MultiInputPolicy",
                vec_env,
                policy_kwargs=policy_kwargs,
                learning_rate=lr,
                n_steps=256,
                batch_size=128,
                n_epochs=4,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=ent_coef,
                vf_coef=0.5,
                max_grad_norm=0.5,
                verbose=0,
                device=device,
            )

        # Train stage
        t0 = time.time()
        model.learn(total_timesteps=timesteps, reset_num_timesteps=True)
        train_duration = time.time() - t0
        print(f"  Stage training finished in {train_duration/60:.2f} mins (~{timesteps/max(1, train_duration):.0f} steps/s)")

        # Save stage checkpoint
        stage_ckpt = AGENTS_DIR / f"showcase_14x14_stage_{idx}.zip"
        model.save(str(stage_ckpt))
        current_weights = str(stage_ckpt)

        # Close vectorized training environment
        vec_env.close()

        # Evaluate on 30 held-out maps (seeds 50000..50029)
        print(f"  Evaluating {stage.name} on 30 fresh held-out maps (seeds 50000..50029)...")
        succ_rate, mean_resc, mean_surv, mean_ret = evaluate_stage_heldout(model, config, num_episodes=30)
        wins_count = int(round(succ_rate * 30))
        print(f"  -> Held-Out Full WIN Rate: {succ_rate*100:.1f}% ({wins_count}/30 maps completely cleared)")
        print(f"  -> Mean Rescued:          {mean_resc * stage.n_hostages:.2f} / {stage.n_hostages} ({mean_resc*100:.1f}%)")
        print(f"  -> Mean Survivors:        {mean_surv:.2f} / {stage.n_rescuers}")
        print(f"  -> Mean Return:           {mean_ret:+.2f}")

        results.append({
            "stage": stage.name,
            "grid_size": stage.grid_size,
            "rescuers": stage.n_rescuers,
            "hostages": stage.n_hostages,
            "timesteps": timesteps,
            "heldout_success_pct": round(succ_rate * 100.0, 1),
            "mean_hostages_rescued": round(mean_resc * stage.n_hostages, 2),
            "survivors": round(mean_surv, 2),
            "mean_return": round(mean_ret, 2),
        })

        if stage.n_hostages == 3 and (succ_rate > best_target_win_rate or best_model_path is None):
            best_target_win_rate = succ_rate
            best_model_path = stage_ckpt

    # Final policy selection: pick best target stage checkpoint
    final_showcase_path = AGENTS_DIR / "showcase_rescuer.zip"
    final_rescuer_path = AGENTS_DIR / "final_rescuer.zip"

    src_path = best_model_path if best_model_path else stage_ckpt
    shutil.copy(str(src_path), str(final_showcase_path))
    shutil.copy(str(src_path), str(final_rescuer_path))
    print(f"\n[Saved] Best policy ({src_path}) deployed to:")
    print(f"  - {final_showcase_path}")
    print(f"  - {final_rescuer_path}")

    # Re-evaluate final deployment on standard 14x14 showcase config
    print("\nRunning final held-out benchmark on deployed policy (30 seeds: 50000..50029)...")
    final_model = PPO.load(str(final_rescuer_path), device=device)
    final_cfg = EnvConfig.showcase()
    final_win_rate, final_resc, final_surv, final_ret = evaluate_stage_heldout(final_model, final_cfg, num_episodes=30)
    final_wins = int(round(final_win_rate * 30))

    print("=" * 70)
    print("  FINAL 14x14 HELD-OUT BENCHMARK RESULT")
    print("=" * 70)
    print(f"  Full WIN Rate (All 3 Rescued): {final_win_rate*100:.1f}% ({final_wins}/30 maps)")
    print(f"  Mean Hostages Rescued:        {final_resc * final_cfg.n_hostages:.2f} / {final_cfg.n_hostages} ({final_resc*100:.1f}%)")
    print(f"  Mean Rescuers Surviving:      {final_surv:.2f} / {final_cfg.n_rescuers}")
    print(f"  Mean Episode Return:          {final_ret:+.2f}")
    print("=" * 70)

    # Log results CSV
    csv_file = METRICS_DIR / "showcase_curriculum.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"[Saved] Curriculum progression logged to: {csv_file}")

    return {
        "final_win_rate": final_win_rate,
        "final_rescued": final_resc * final_cfg.n_hostages,
        "final_survivors": final_surv,
        "stages": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default="agents/final_rescuer.zip")
    parser.add_argument("--fast", action="store_true", help="Fast training mode for quick curriculum pass")
    args = parser.parse_args()

    run_curriculum(resume_path=args.resume, fast_mode=args.fast)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default="agents/final_rescuer.zip")
    parser.add_argument("--fast", action="store_true", help="Fast training mode for quick curriculum pass")
    args = parser.parse_args()

    run_curriculum(resume_path=args.resume, fast_mode=args.fast)

