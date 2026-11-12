"""Day 10 Difficulty Preset Balance Sweep and Curve Generator.

Evaluates rescuer-vs-adversary matchups on 30 held-out maze seeds (50000-50029)
across balance knobs to identify:
- Balanced preset: ~50% rescuer full-win rate (acceptance target: 45-55%)
- Easy preset: ~70% rescuer full-win rate
- Brutal preset: ~30% rescuer full-win rate

Generates publication-quality balance curve at metrics/day10_balance_curve.png.
"""

import argparse
import csv
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import PPO

from env.config import EnvConfig
from env.fire_controller import PolicyFireController
from env.rescue_env import RadonRescueEnv


def evaluate_preset(
    rescuer_model: PPO,
    fire_controller: PolicyFireController,
    config: EnvConfig,
    num_episodes: int = 30,
    base_seed: int = 50000,
) -> Dict[str, Any]:
    """Runs evaluation on held-out seeds with batched rescuer inference."""
    full_wins = 0
    partial_wins = 0
    rescued_list: List[int] = []
    survivors_list: List[int] = []
    deaths_list: List[int] = []
    hp_lost_list: List[int] = []
    dashes_list: List[int] = []
    lengths_list: List[int] = []

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
            carry_capacity=config.carry_capacity,
            carry_overtime=config.carry_overtime,
            obs_bfs_features=config.obs_bfs_features,
            seed=ep_seed,
            layout_type=config.layout_type,
            rescuer_hp=config.rescuer_hp,
            invulnerability_steps=config.invulnerability_steps,
            fire_lifetime=config.fire_lifetime,
            ash_cooldown=config.ash_cooldown,
            fire_ignitions_per_step=config.fire_ignitions_per_step,
            no_point_blank_ignition=config.no_point_blank_ignition,
        )

        env = RadonRescueEnv(config=ep_cfg, fire_controller=fire_controller)
        obs, _ = env.reset(seed=ep_seed)

        step_count = 0
        while env.agents and step_count < config.max_steps:
            step_count += 1
            actions = {}
            living_agents = [aid for aid in env.agents if env.rescuers[aid].alive]
            if living_agents:
                batch_obs = {
                    k: np.stack([obs[aid][k] for aid in living_agents], axis=0)
                    for k in obs[living_agents[0]].keys()
                }
                acts, _ = rescuer_model.predict(batch_obs, deterministic=False)
                for aid, act in zip(living_agents, acts):
                    actions[aid] = int(act)
            obs, rews, terms, truncs, infos = env.step(actions)

        rescued = sum(1 for h in env.hostages if h.rescued)
        rescued_list.append(rescued)
        if rescued == config.n_hostages:
            full_wins += 1
        if rescued > 0:
            partial_wins += 1

        survivors = sum(1 for r in env.rescuers.values() if r.alive)
        survivors_list.append(survivors)
        deaths_list.append(getattr(env, "fire_deaths", config.n_rescuers - survivors))
        hp_lost_list.append(getattr(env, "hp_lost", 0))
        dashes_list.append(getattr(env, "fire_dashes_survived", 0))
        lengths_list.append(env.current_step)

    return {
        "full_win_rate": round(full_wins / num_episodes * 100.0, 1),
        "partial_win_rate": round(partial_wins / num_episodes * 100.0, 1),
        "mean_rescued": round(float(np.mean(rescued_list)), 2),
        "rescued_pct": round(float(np.mean(rescued_list)) / config.n_hostages * 100.0, 1),
        "mean_survivors": round(float(np.mean(survivors_list)), 2),
        "mean_deaths": round(float(np.mean(deaths_list)), 2),
        "mean_hp_lost": round(float(np.mean(hp_lost_list)), 2),
        "mean_dashes_survived": round(float(np.mean(dashes_list)), 2),
        "mean_length": round(float(np.mean(lengths_list)), 1),
    }


def plot_balance_curve(
    results: List[Dict[str, Any]],
    output_path: str = "metrics/day10_balance_curve.png",
) -> None:
    """Generates a publication-quality 2-panel figure showing the balance curve across presets."""
    preset_names = [r["preset"] for r in results]
    win_rates = [r["full_win_rate"] for r in results]
    partial_rates = [r["partial_win_rate"] for r in results]
    rescued = [r["mean_rescued"] for r in results]
    deaths = [r["mean_deaths"] for r in results]
    hp_lost = [r["mean_hp_lost"] for r in results]
    dashes = [r["mean_dashes_survived"] for r in results]

    x = np.arange(len(preset_names))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), dpi=150)
    plt.subplots_adjust(wspace=0.28)
    grid_kw = dict(color="#E0E0E0", linestyle="--", linewidth=0.7, alpha=0.8)

    # Panel 1: Win Rates & Evacuation
    ax1.plot(x, win_rates, color="#16a34a", marker="o", markersize=9, linewidth=2.5, label="Full Win Rate (%)")
    ax1.plot(x, partial_rates, color="#2563eb", marker="s", markersize=8, linewidth=2.2, linestyle="--", label="Partial Win Rate (%)")
    ax1.axhline(50.0, color="#f59e0b", linestyle=":", linewidth=2.0, label="50% Balanced Target")
    ax1.axhspan(45.0, 55.0, color="#fef3c7", alpha=0.5, label="Fair Target Band (45-55%)")

    # Annotate presets
    for idx, (p, w) in enumerate(zip(preset_names, win_rates)):
        ax1.annotate(f"{w:.1f}%", (idx, w + 3.0), ha="center", fontsize=9.5, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(preset_names, fontsize=10, fontweight="bold")
    ax1.set_ylabel("Win Rate (%)", fontsize=11, fontweight="bold")
    ax1.set_ylim(-5, 105)
    ax1.grid(True, **grid_kw)
    ax1.set_title("Rescuer Performance Across Difficulty Presets", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right", framealpha=0.9)

    # Panel 2: Combat & Hazard Dynamics
    ax2.plot(x, rescued, color="#059669", marker="^", markersize=8, linewidth=2.5, label="Hostages Rescued (/4)")
    ax2.plot(x, deaths, color="#dc2626", marker="x", markersize=8, linewidth=2.2, label="Fire Fatalities (/5)")
    ax2.plot(x, dashes, color="#7c3aed", marker="D", markersize=7, linewidth=2.0, linestyle="-.", label="Fire Dashes Survived")

    ax2.set_xticks(x)
    ax2.set_xticklabels(preset_names, fontsize=10, fontweight="bold")
    ax2.set_ylabel("Count per Episode", fontsize=11, fontweight="bold")
    ax2.set_ylim(-0.2, 4.8)
    ax2.grid(True, **grid_kw)
    ax2.set_title("Evacuation Volume vs Combat Hazard Impact", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper right", framealpha=0.9)

    fig.suptitle("RADON Day 10 — Balance Curve & Preset Verification on Held-Out Mazes", fontsize=14, fontweight="bold", y=1.02)
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_p, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved balance curve to {out_p}")


def run_sweep(
    rescuer_path: str = "agents/day10_rescuer_r2.zip",
    fire_path: str = "agents/day10_fire_r2.zip",
    num_episodes: int = 30,
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Starting Day 10 Preset Balance Sweep (Device: {device}) ===")

    # Fallback to R0 / R1 if R2 paths are not available yet
    if not Path(rescuer_path).exists():
        rescuer_path = "agents/day10_rescuer_r0.zip"
    if not Path(fire_path).exists():
        fire_path = "agents/day10_fire_r1.zip"

    print(f"Loading Rescuer from: {rescuer_path}")
    rescuer_model = PPO.load(rescuer_path, device=device)

    print(f"Loading Fire Adversary from: {fire_path}")
    fire_model = PPO.load(fire_path, device=device)
    fire_controller = PolicyFireController(model=fire_model, deterministic=False)

    # Candidate Presets to test:
    # 1. Easy: shorter fire lifetime (22 steps), lower fire cap (32), faster burnout -> ~70% win rate
    # 2. Balanced: standard maze tunings (lifetime 28, max_tiles 45, rescuer_hp 2) -> ~50% target
    # 3. Brutal: persistent fire (lifetime 36, max_tiles 55, faster spread) -> ~30% win rate
    presets = [
        {
            "name": "Easy",
            "fire_lifetime": 14,
            "fire_max_tiles": 18,
            "fire_spread_prob": 0.005,
            "fire_ignitions_per_step": 1,
            "rescuer_hp": 2,
        },
        {
            "name": "Balanced",
            "fire_lifetime": 16,
            "fire_max_tiles": 22,
            "fire_spread_prob": 0.006,
            "fire_ignitions_per_step": 1,
            "rescuer_hp": 2,
        },
        {
            "name": "Brutal",
            "fire_lifetime": 24,
            "fire_max_tiles": 35,
            "fire_spread_prob": 0.010,
            "fire_ignitions_per_step": 1,
            "rescuer_hp": 2,
        },
    ]

    all_results = []
    for p in presets:
        print(f"\n--- Testing Preset: {p['name']} ---")
        cfg = EnvConfig.maze(
            grid_size=20,
            fire_lifetime=p["fire_lifetime"],
            fire_max_tiles=p["fire_max_tiles"],
            fire_ignitions_per_step=p["fire_ignitions_per_step"],
            rescuer_hp=p["rescuer_hp"],
        )
        cfg.fire_spread_prob = p["fire_spread_prob"]

        res = evaluate_preset(
            rescuer_model=rescuer_model,
            fire_controller=fire_controller,
            config=cfg,
            num_episodes=num_episodes,
        )
        res["preset"] = p["name"]
        res.update(p)
        all_results.append(res)
        print(f"[{p['name']}] Win Rate: {res['full_win_rate']}%, Rescued: {res['mean_rescued']}/4, "
              f"Deaths: {res['mean_deaths']}, Dashes: {res['mean_dashes_survived']}, HP Lost: {res['mean_hp_lost']}")

    # Save to CSV
    csv_file = Path("metrics/day10_balance_curve.csv")
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(all_results[0].keys())
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            writer.writerow(r)
    print(f"\n[CSV] Saved balance results to {csv_file}")

    # Plot balance curve
    plot_balance_curve(all_results, output_path="metrics/day10_balance_curve.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rescuer", type=str, default="agents/day10_rescuer_r2.zip")
    parser.add_argument("--fire", type=str, default="agents/day10_fire_r2.zip")
    parser.add_argument("--episodes", type=int, default=30)
    args = parser.parse_args()

    run_sweep(
        rescuer_path=args.rescuer,
        fire_path=args.fire,
        num_episodes=args.episodes,
    )
