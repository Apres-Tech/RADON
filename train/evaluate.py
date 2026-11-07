"""Evaluation suite comparing trained PPO policy against random baseline and exporting replay."""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
from stable_baselines3 import PPO

from env.config import EnvConfig
from env.replay import EpisodeRecorder
from env.rescue_env import RadonRescueEnv


def run_episodes(
    env: RadonRescueEnv,
    model: PPO | None,
    num_episodes: int,
    base_seed: int,
    deterministic: bool = False,
    record_first_to: Path | None = None,
    vary_seeds: bool = False,
) -> Dict[str, float]:
    """Runs evaluation episodes with deterministic/stochastic policy or random actions."""
    returns = []
    rescued_counts = []
    successes = []
    surviving_counts = []
    episode_lengths = []

    for ep_idx in range(num_episodes):
        seed = (base_seed + ep_idx) if vary_seeds else base_seed
        obs, infos = env.reset(seed=seed)

        recorder = None
        if ep_idx == 0 and record_first_to is not None:
            recorder = EpisodeRecorder(env)
            recorder.record_initial_frame()

        ep_return = 0.0

        while env.agents:
            actions = {}
            for agent_id in env.agents:
                if model is not None:
                    agent_obs = {
                        k: np.expand_dims(v, axis=0) for k, v in obs[agent_id].items()
                    }
                    action, _ = model.predict(agent_obs, deterministic=deterministic)
                    actions[agent_id] = int(action[0])
                else:
                    actions[agent_id] = int(env.action_space(agent_id).sample())

            obs, rewards, terminations, truncations, infos = env.step(actions)
            if recorder is not None:
                recorder.record_step(rewards)
            ep_return += sum(rewards.values())

        if recorder is not None and record_first_to is not None:
            recorder.save(record_first_to)

        rescued = sum(1 for h in env.hostages if h.rescued)
        success = 1.0 if rescued == env.config.n_hostages else 0.0
        surviving = sum(1 for r in env.rescuers.values() if r.alive)

        returns.append(ep_return)
        rescued_counts.append(rescued)
        successes.append(success)
        surviving_counts.append(surviving)
        episode_lengths.append(env.current_step)

    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_rescued": float(np.mean(rescued_counts)),
        "success_rate": float(np.mean(successes)),
        "mean_surviving": float(np.mean(surviving_counts)),
        "mean_length": float(np.mean(episode_lengths)),
    }


def compare_and_report(
    model_path: str | Path,
    num_episodes: int = 100,
    base_seed: int = 42,
    deterministic: bool = False,
    replay_out: Path | str = "replays/day2_trained.json",
    vary_seeds: bool = False,
):
    """Loads trained model, evaluates against random policy, prints table, exports replay."""
    config = EnvConfig.training_day2(seed=base_seed)
    env = RadonRescueEnv(config=config)

    # 1. Evaluate Random Policy
    mode_str = "fresh randomized maps" if vary_seeds else f"map seed {base_seed}"
    print(f"\n[1/2] Evaluating Random Policy over {num_episodes} episodes ({mode_str})...")
    random_metrics = run_episodes(
        env,
        model=None,
        num_episodes=num_episodes,
        base_seed=base_seed,
        vary_seeds=vary_seeds,
    )

    # 2. Evaluate Trained PPO Policy
    print(f"[2/2] Evaluating Trained PPO Policy ({model_path}) over {num_episodes} episodes (deterministic={deterministic}, {mode_str})...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PPO.load(model_path, device=device)
    replay_path = Path(replay_out)
    replay_path.parent.mkdir(parents=True, exist_ok=True)
    trained_metrics = run_episodes(
        env,
        model=model,
        num_episodes=num_episodes,
        base_seed=base_seed,
        deterministic=deterministic,
        record_first_to=replay_path,
        vary_seeds=vary_seeds,
    )

    env.close()

    # 3. Print Side-by-Side Comparison Table
    print("\n" + "=" * 76)
    print(f"  RADON Day 2 Policy Evaluation Benchmark ({num_episodes} Episodes)")
    print("=" * 76)
    header = f"{'Metric':<32} | {'Random Baseline':<18} | {'Trained PPO (Day 2)':<18}"
    print(header)
    print("-" * 76)
    rows = [
        ("Mean Hostages Rescued (/3)", f"{random_metrics['mean_rescued']:4.2f} / 3", f"{trained_metrics['mean_rescued']:4.2f} / 3"),
        ("Success Rate (All Rescued)", f"{random_metrics['success_rate']*100:5.1f}%", f"{trained_metrics['success_rate']*100:5.1f}%"),
        ("Mean Episode Return", f"{random_metrics['mean_return']:+7.2f} +/- {random_metrics['std_return']:.1f}", f"{trained_metrics['mean_return']:+7.2f} +/- {trained_metrics['std_return']:.1f}"),
        ("Mean Rescuers Surviving (/4)", f"{random_metrics['mean_surviving']:4.2f} / 4", f"{trained_metrics['mean_surviving']:4.2f} / 4"),
        ("Mean Episode Length (steps)", f"{random_metrics['mean_length']:5.1f} steps", f"{trained_metrics['mean_length']:5.1f} steps"),
    ]
    for label, r_val, t_val in rows:
        print(f"{label:<32} | {r_val:<18} | {t_val:<18}")
    print("=" * 76)

    print(f"\n[Export] Replay saved to: {replay_path}")

    # Check requirement: Trained MUST beat random on hostages-rescued
    diff = trained_metrics["mean_rescued"] - random_metrics["mean_rescued"]
    if diff > 0:
        print(f"[VERIFIED] Trained policy beats random baseline by +{diff:.2f} mean hostages rescued!")
    else:
        print(f"[WARNING] Trained policy did not beat random baseline (diff={diff:.2f}). Tuning needed.")

    return random_metrics, trained_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RADON Day 2 Policy")
    parser.add_argument(
        "--model-path",
        type=str,
        default="agents/ppo_day2_rescue_team.zip",
        help="Path to saved model checkpoint",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes (default: 100)",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=42,
        help="Base seed for evaluation episodes (default: 42)",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic mode instead of stochastic sampling",
    )
    parser.add_argument(
        "--fresh-maps",
        action="store_true",
        help="Vary map procedural seeds across episodes (default: False)",
    )
    parser.add_argument(
        "--replay-out",
        type=str,
        default="replays/day2_5_trained.json",
        help="Path for exported replay JSON",
    )
    args = parser.parse_args()
    compare_and_report(
        args.model_path,
        args.episodes,
        args.base_seed,
        deterministic=args.deterministic,
        replay_out=args.replay_out,
        vary_seeds=args.fresh_maps,
    )
