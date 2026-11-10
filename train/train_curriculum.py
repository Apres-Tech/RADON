"""Progressive curriculum trainer for RADON large open showcase environments (12x12 -> 16x16 -> 20x20)."""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import time
from typing import List, Tuple

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
        name="Stage 1 (12x12)",
        grid_size=12,
        n_rescuers=3,
        n_hostages=2,
        max_steps=180,
        fire_max_tiles=25,
        timesteps=45_000,
        num_vec_envs=6,
    ),
    CurriculumStage(
        name="Stage 2 (16x16)",
        grid_size=16,
        n_rescuers=4,
        n_hostages=3,
        max_steps=250,
        fire_max_tiles=40,
        timesteps=50_000,
        num_vec_envs=4,
    ),
    CurriculumStage(
        name="Stage 3 (20x20)",
        grid_size=20,
        n_rescuers=5,
        n_hostages=4,
        max_steps=350,
        fire_max_tiles=60,
        timesteps=60_000,
        num_vec_envs=4,
    ),
]


def evaluate_stage_heldout(
    model: PPO,
    config: EnvConfig,
    num_episodes: int = 25,
    base_seed: int = 50000,
) -> Tuple[float, float, float, float]:
    """Evaluates policy over fixed held-out unseen maps."""
    successes = []
    rescued_ratios = []
    surviving_counts = []
    returns = []

    for ep in range(num_episodes):
        ep_seed = base_seed + ep
        ep_cfg = EnvConfig(
            floors=config.floors,
            grid_h=config.grid_h,
            grid_w=config.grid_w,
            n_rescuers=config.n_rescuers,
            n_hostages=config.n_hostages,
            max_steps=config.max_steps,
            view_radius=config.view_radius,
            fire_spread_prob=config.fire_spread_prob,
            fire_spread_start_step=config.fire_spread_start_step,
            fire_max_tiles=config.fire_max_tiles,
            fire_enabled=config.fire_enabled,
            seed=ep_seed,
            enable_shaping=config.enable_shaping,
            shaping_coef=config.shaping_coef,
        )

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
    print("  RADON Progressive Showcase Curriculum Trainer")
    print(f"  Device: {device} | Base Model: {resume_path}")
    print("=" * 70)

    AGENTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    current_weights = resume_path if Path(resume_path).exists() else "agents/base_frozen.zip"
    results = []

    for idx, stage in enumerate(STAGES, 1):
        timesteps = stage.timesteps if not fast_mode else stage.timesteps // 2
        print(f"\n[{idx}/3] Starting {stage.name} — Grid: {stage.grid_size}x{stage.grid_size}, Rescuers: {stage.n_rescuers}, Hostages: {stage.n_hostages}, Timesteps: {timesteps:,}")

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

        if current_weights and Path(current_weights).exists():
            print(f"  Loading pretrained weights from: {current_weights}")
            model = PPO.load(
                current_weights,
                env=vec_env,
                learning_rate=3e-4,
                ent_coef=0.03,
                device=device,
            )
        else:
            model = PPO(
                "MultiInputPolicy",
                vec_env,
                policy_kwargs=policy_kwargs,
                learning_rate=3e-4,
                n_steps=256,
                batch_size=128,
                n_epochs=4,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.03,
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
        stage_ckpt = AGENTS_DIR / f"showcase_stage_{idx}.zip"
        model.save(str(stage_ckpt))
        current_weights = str(stage_ckpt)

        # Evaluate on held-out maps
        print(f"  Evaluating {stage.name} on 25 fresh held-out maps (seeds 50000..50024)...")
        succ_rate, mean_resc, mean_surv, mean_ret = evaluate_stage_heldout(model, config, num_episodes=25)
        print(f"  -> Held-Out Success Rate: {succ_rate*100:.1f}%")
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

    # Save final showcase model
    final_showcase_path = AGENTS_DIR / "showcase_rescuer.zip"
    model.save(str(final_showcase_path))
    print(f"\n[Saved] Final showcase policy saved to: {final_showcase_path}")

    # Log results CSV
    csv_file = METRICS_DIR / "showcase_curriculum.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"[Saved] Curriculum progression logged to: {csv_file}")

    print("\n" + "=" * 70)
    print("  RADON Curriculum Retraining Summary")
    print("=" * 70)
    for r in results:
        print(f"  {r['stage']}: Success {r['heldout_success_pct']}% | Rescued {r['mean_hostages_rescued']}/{r['hostages']} | Survivors {r['survivors']}/{r['rescuers']}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default="agents/final_rescuer.zip")
    parser.add_argument("--fast", action="store_true", help="Fast training mode for quick curriculum pass")
    args = parser.parse_args()

    run_curriculum(resume_path=args.resume, fast_mode=args.fast)

