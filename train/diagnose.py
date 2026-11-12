"""Failure-mode and navigation diagnostics for RADON rescue agents.

Analyzes why episodes succeed or fail on held-out maps:
- Full win rate, partial rescue rate, timeouts, wipes
- Idle / oscillation / revisiting rate (blind wandering metric)
- Reachability of unrescued hostages via BFS at termination
- End BFS distance from free living rescuers to unrescued hostages
"""

import argparse
from collections import deque
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from stable_baselines3 import PPO

from env.config import EnvConfig
from env.fire_controller import CurriculumFireController
from env.rescue_env import RadonRescueEnv, Tile


def compute_bfs_reachability_and_distance(
    env: RadonRescueEnv,
) -> Tuple[float, float, int]:
    """Calculates reachability of unrescued hostages and mean distance from free rescuers.

    Returns:
        (reachability_ratio, mean_distance_to_hostage, unrescued_count)
    """
    unrescued = [h for h in env.hostages if not h.rescued and h.carried_by is None]
    if not unrescued:
        return 1.0, 0.0, 0

    # Targets for BFS: exits and alive rescuers
    bases: List[Tuple[int, int, int]] = []
    if env.building_map:
        bases.extend(env.building_map.exits)
    for r in env.rescuers.values():
        if r.alive:
            bases.append((r.floor, r.row, r.col))

    base_dist_map = env._compute_multi_floor_bfs_dist_map(bases)
    reachable_count = 0
    for h in unrescued:
        if base_dist_map[h.floor, h.row, h.col] < 900:
            reachable_count += 1
    reachability_ratio = reachable_count / len(unrescued)

    # Compute distance from living free rescuers to nearest unrescued hostage
    hostage_targets = [(h.floor, h.row, h.col) for h in unrescued]
    hostage_dist_map = env._compute_multi_floor_bfs_dist_map(hostage_targets)

    free_rescuers = [r for r in env.rescuers.values() if r.alive and not r.carrying]
    if free_rescuers:
        dists = [
            float(hostage_dist_map[r.floor, r.row, r.col])
            for r in free_rescuers
        ]
        # Cap unreachable distances at 2 * (grid_h + grid_w) for fair averaging
        max_dist_cap = 2.0 * (env.config.grid_h + env.config.grid_w)
        capped_dists = [min(d, max_dist_cap) for d in dists]
        mean_dist = float(np.mean(capped_dists))
    else:
        mean_dist = float(2.0 * (env.config.grid_h + env.config.grid_w))

    return reachability_ratio, mean_dist, len(unrescued)


def run_diagnostics(
    model_path: Optional[str] = None,
    num_episodes: int = 30,
    base_seed: int = 50000,
    preset: str = "showcase",
    deterministic: bool = False,
    window_revisit: int = 10,
    model_obj: Any = None,
) -> Dict[str, Any]:
    """Runs diagnostic evaluation over held-out seeds."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model_obj
    if model is None and model_path is not None and Path(model_path).exists():
        # Check if sb3_contrib RecurrentPPO
        loaded = False
        try:
            from sb3_contrib import RecurrentPPO
            model = RecurrentPPO.load(model_path, device=device)
            loaded = True
        except Exception:
            pass
        if not loaded:
            model = PPO.load(model_path, device=device)

    # Auto-detect whether model expects 6 or 13 self-features
    obs_bfs_features = True
    if model is not None and hasattr(model, "observation_space"):
        try:
            self_space = model.observation_space["self"]
            if self_space.shape[0] == 6:
                obs_bfs_features = False
        except Exception:
            pass

    # Resolve config
    if preset == "showcase":
        cfg_template = EnvConfig.showcase(obs_bfs_features=obs_bfs_features)
    elif preset == "showcase_hard":
        cfg_template = EnvConfig.showcase_hard(obs_bfs_features=obs_bfs_features)
    elif preset == "training_day4_base":
        cfg_template = EnvConfig.training_day4_base(obs_bfs_features=obs_bfs_features)
    else:
        cfg_template = EnvConfig.showcase(obs_bfs_features=obs_bfs_features)

    # Track metrics
    full_wins = 0
    timeouts = 0
    wipes = 0
    rescued_counts: List[int] = []
    surviving_counts: List[int] = []
    fire_deaths_counts: List[int] = []
    episode_lengths: List[int] = []

    total_agent_steps = 0
    revisited_agent_steps = 0

    end_reachabilities: List[float] = []
    end_distances: List[float] = []

    # LSTM state tracking per agent if recurrent
    is_recurrent = hasattr(model, "policy") and hasattr(model.policy, "lstm")

    for ep in range(num_episodes):
        ep_seed = base_seed + ep
        if preset == "showcase":
            cfg = EnvConfig.showcase(seed=ep_seed, obs_bfs_features=obs_bfs_features)
        elif preset == "showcase_hard":
            cfg = EnvConfig.showcase_hard(seed=ep_seed, obs_bfs_features=obs_bfs_features)
        else:
            cfg = EnvConfig.training_day4_base(seed=ep_seed, obs_bfs_features=obs_bfs_features)

        env = RadonRescueEnv(config=cfg, fire_controller=CurriculumFireController())
        obs, _ = env.reset(seed=ep_seed)

        # Position history for each rescuer (last window_revisit positions)
        pos_history: Dict[str, deque] = {
            aid: deque(maxlen=window_revisit) for aid in env.agents
        }
        lstm_states: Dict[str, Any] = {aid: None for aid in env.agents} if is_recurrent else {}
        ep_starts: Dict[str, np.ndarray] = {
            aid: np.ones((1,), dtype=bool) for aid in env.agents
        } if is_recurrent else {}

        while env.agents:
            actions = {}
            for aid in env.agents:
                rescuer = env.rescuers[aid]
                if rescuer.alive:
                    cur_pos = (rescuer.floor, rescuer.row, rescuer.col)
                    total_agent_steps += 1
                    if cur_pos in pos_history[aid]:
                        revisited_agent_steps += 1
                    pos_history[aid].append(cur_pos)

                if model is not None:
                    agent_obs = {
                        k: np.expand_dims(v, axis=0) for k, v in obs[aid].items()
                    }
                    if is_recurrent:
                        act, lstm_states[aid] = model.predict(
                            agent_obs,
                            state=lstm_states[aid],
                            episode_start=ep_starts[aid],
                            deterministic=deterministic,
                        )
                        ep_starts[aid] = np.zeros((1,), dtype=bool)
                    else:
                        act, _ = model.predict(agent_obs, deterministic=deterministic)
                    actions[aid] = int(act[0])
                else:
                    actions[aid] = int(env.action_space(aid).sample())

            obs, rewards, terms, truncs, _ = env.step(actions)

        # End of episode evaluation
        verdict = env.verdict
        if verdict == "WIN":
            full_wins += 1
        elif verdict == "LOSE_WIPED":
            wipes += 1
        else:
            timeouts += 1

        rescued = sum(1 for h in env.hostages if h.rescued)
        rescued_counts.append(rescued)
        survivors = sum(1 for r in env.rescuers.values() if r.alive)
        surviving_counts.append(survivors)
        fire_deaths_counts.append(getattr(env, "fire_deaths", cfg.n_rescuers - survivors))
        episode_lengths.append(env.current_step)

        # Reachability & end distance diagnostics
        reach_ratio, mean_d, unrescued_cnt = compute_bfs_reachability_and_distance(env)
        end_reachabilities.append(reach_ratio)
        if unrescued_cnt > 0:
            end_distances.append(mean_d)

    # Aggregate
    oscillation_rate = (
        (revisited_agent_steps / total_agent_steps * 100.0)
        if total_agent_steps > 0
        else 0.0
    )
    full_win_rate = full_wins / num_episodes * 100.0
    partial_rate = (
        sum(1 for r in rescued_counts if r > 0) / num_episodes * 100.0
    )
    mean_rescued = float(np.mean(rescued_counts))
    max_h = cfg_template.n_hostages
    mean_reachability = float(np.mean(end_reachabilities)) * 100.0
    mean_end_dist = float(np.mean(end_distances)) if end_distances else 0.0

    results = {
        "num_episodes": num_episodes,
        "full_wins": full_wins,
        "full_win_rate": full_win_rate,
        "partial_rate": partial_rate,
        "timeouts": timeouts,
        "wipes": wipes,
        "mean_rescued": mean_rescued,
        "max_hostages": max_h,
        "oscillation_rate": oscillation_rate,
        "revisited_agent_steps": revisited_agent_steps,
        "total_agent_steps": total_agent_steps,
        "reachability_rate": mean_reachability,
        "mean_end_distance": mean_end_dist,
        "mean_surviving": float(np.mean(surviving_counts)),
        "mean_fire_deaths": float(np.mean(fire_deaths_counts)),
        "mean_length": float(np.mean(episode_lengths)),
    }

    return results


def print_diagnostic_table(title: str, res: Dict[str, Any]) -> None:
    """Formats and prints diagnostic results table."""
    print("\n" + "=" * 76)
    print(f"  {title}")
    print("=" * 76)
    rows = [
        ("Held-out Episodes Evaluated", f"{res['num_episodes']}"),
        ("Full Win Rate (All Evacuated)", f"{res['full_win_rate']:5.1f}% ({res['full_wins']}/{res['num_episodes']})"),
        ("Partial Rescue Rate (>=1 Saved)", f"{res['partial_rate']:5.1f}%"),
        ("Mean Hostages Rescued", f"{res['mean_rescued']:4.2f} / {res['max_hostages']} ({res['mean_rescued']/res['max_hostages']*100:4.1f}%)"),
        ("Outcomes: Timeouts / Wipes", f"{res['timeouts']} timeouts / {res['wipes']} wiped"),
        ("Idle/Oscillation (Revisit <=10 steps)", f"{res['oscillation_rate']:5.1f}% ({res['revisited_agent_steps']}/{res['total_agent_steps']} steps)"),
        ("Unrescued Reachability at Horizon", f"{res['reachability_rate']:5.1f}%"),
        ("Mean BFS Distance to Nearest Hostage", f"{res['mean_end_distance']:4.2f} tiles"),
        ("Team Survival Rate", f"{res['mean_surviving']:.2f} alive (deaths: {res['mean_fire_deaths']:.2f})"),
        ("Mean Episode Length", f"{res['mean_length']:.1f} steps"),
    ]
    for label, val in rows:
        print(f"  {label:<38} | {val}")
    print("=" * 76)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RADON Failure-Mode Diagnostics")
    parser.add_argument(
        "--model-path",
        type=str,
        default="agents/showcase_rescuer.zip",
        help="Path to trained model (.zip)",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="showcase",
        choices=["showcase", "showcase_hard", "training_day4_base"],
        help="Environment preset",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=30,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=50000,
        help="Base seed for held-out evaluation",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic policy mode",
    )
    args = parser.parse_args()

    title = f"RADON Failure Diagnostics — Model: {Path(args.model_path).name} (Held-out seeds {args.base_seed}..{args.base_seed+args.episodes-1})"
    results = run_diagnostics(
        model_path=args.model_path,
        num_episodes=args.episodes,
        base_seed=args.base_seed,
        preset=args.preset,
        deterministic=args.deterministic,
    )
    print_diagnostic_table(title, results)
