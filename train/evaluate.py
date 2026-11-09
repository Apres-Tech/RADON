"""Evaluation suite comparing trained PPO policy against random baseline and exporting replay."""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
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
    fire_deaths_counts = []
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
        f_deaths = getattr(env, "fire_deaths", env.config.n_rescuers - surviving)

        returns.append(ep_return)
        rescued_counts.append(rescued)
        successes.append(success)
        surviving_counts.append(surviving)
        fire_deaths_counts.append(f_deaths)
        episode_lengths.append(env.current_step)

    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_rescued": float(np.mean(rescued_counts)),
        "success_rate": float(np.mean(successes)),
        "mean_surviving": float(np.mean(surviving_counts)),
        "mean_fire_deaths": float(np.mean(fire_deaths_counts)),
        "mean_length": float(np.mean(episode_lengths)),
    }


def compare_and_report(
    model_path: str | Path,
    num_episodes: int = 100,
    base_seed: int = 42,
    deterministic: bool = False,
    replay_out: Path | str = "replays/day3_trained.json",
    vary_seeds: bool = False,
    preset: str = "training_day4_base",
    fire_model: Optional[str] = None,
):
    """Loads trained model, evaluates against random baseline, prints table, exports replay."""
    if preset == "sanity":
        config = EnvConfig.sanity(seed=base_seed)
    elif preset == "training_day4_base":
        config = EnvConfig.training_day4_base(seed=base_seed)
    elif preset == "training_day3":
        config = EnvConfig.training_day3(seed=base_seed)
    else:
        config = EnvConfig.training_day2(seed=base_seed)

    fire_ctrl = None
    if fire_model and Path(fire_model).exists():
        from env.fire_controller import PolicyFireController
        fire_ctrl = PolicyFireController(model_path=fire_model)

    env = RadonRescueEnv(config=config, fire_controller=fire_ctrl)

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
    max_h = config.n_hostages
    print("\n" + "=" * 76)
    print(f"  RADON Day 3 Policy Evaluation Benchmark ({num_episodes} Episodes)")
    print("=" * 76)
    header = f"{'Metric':<32} | {'Random Baseline':<18} | {'Trained PPO (Day 3)':<18}"
    print(header)
    print("-" * 76)
    rows = [
        (f"Mean Hostages Rescued (/{max_h})", f"{random_metrics['mean_rescued']:4.2f} / {max_h}", f"{trained_metrics['mean_rescued']:4.2f} / {max_h}"),
        ("Success Rate (All Rescued)", f"{random_metrics['success_rate']*100:5.1f}%", f"{trained_metrics['success_rate']*100:5.1f}%"),
        ("Mean Episode Return", f"{random_metrics['mean_return']:+7.2f} +/- {random_metrics['std_return']:.1f}", f"{trained_metrics['mean_return']:+7.2f} +/- {trained_metrics['std_return']:.1f}"),
        (f"Mean Rescuers Surviving (/{config.n_rescuers})", f"{random_metrics['mean_surviving']:4.2f} / {config.n_rescuers}", f"{trained_metrics['mean_surviving']:4.2f} / {config.n_rescuers}"),
        ("Mean Fire Deaths", f"{random_metrics['mean_fire_deaths']:4.2f}", f"{trained_metrics['mean_fire_deaths']:4.2f}"),
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
    parser = argparse.ArgumentParser(description="Evaluate RADON Day 3 Policy")
    parser.add_argument(
        "--model-path",
        type=str,
        default="agents/ppo_day3_two_floors.zip",
        help="Path to saved model checkpoint",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="training_day4_base",
        choices=["training_day4_base", "training_day3", "training_day2", "sanity"],
        help="Environment preset (default: training_day4_base)",
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
        default=50000,
        help="Base seed for evaluation episodes (default: 50000 held-out)",
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
        default="replays/day4_adversarial.json",
        help="Path for exported replay JSON",
    )
    parser.add_argument(
        "--fire-model",
        type=str,
        default=None,
        help="Path to trained fire adversary model to evaluate against (default: None)",
    )
    args = parser.parse_args()
    compare_and_report(
        args.model_path,
        args.episodes,
        args.base_seed,
        deterministic=args.deterministic,
        replay_out=args.replay_out,
        vary_seeds=args.fresh_maps,
        preset=args.preset,
        fire_model=args.fire_model,
    )
