"""Random agent rollout and render sanity check for RADON."""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pettingzoo.test

from env.config import EnvConfig
from env.replay import EpisodeRecorder
from env.rescue_env import RadonRescueEnv
from env.tiles import Tile


def render_floors_to_png(env: RadonRescueEnv, output_dir: Path) -> list[Path]:
    """Renders visual 2D floor plans with entities and saves to PNGs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    h, w = env.config.grid_h, env.config.grid_w
    saved_paths = []

    # Color palette
    colors = {
        Tile.EMPTY: "#ECEFF1",  # Light grey floor
        Tile.WALL: "#263238",   # Dark charcoal wall
        Tile.STAIRS: "#7E57C2", # Violet stairs
        Tile.EXIT: "#00E676",   # Green exit
        Tile.HOSTAGE: "#FFB300",# Golden amber hostage
        Tile.FIRE: "#FF1744",   # Vivid red fire
    }

    for f in range(env.config.floors):
        fig, ax = plt.subplots(figsize=(8, 8), dpi=120)
        ax.set_xlim(-0.5, w - 0.5)
        ax.set_ylim(h - 0.5, -0.5)  # Inverted y-axis so row 0 is top
        ax.set_aspect("equal")

        # Draw base tiles
        for r in range(h):
            for c in range(w):
                t = Tile(env.grids[f, r, c])
                rect = plt.Rectangle(
                    (c - 0.5, r - 0.5),
                    1.0,
                    1.0,
                    facecolor=colors[t],
                    edgecolor="#B0BEC5",
                    linewidth=0.5,
                )
                ax.add_patch(rect)
                if t == Tile.STAIRS:
                    ax.text(c, r, "STAIRS", ha="center", va="center", fontsize=7, color="white", weight="bold")
                elif t == Tile.EXIT:
                    ax.text(c, r, "EXIT", ha="center", va="center", fontsize=7, color="black", weight="bold")
                elif t == Tile.FIRE:
                    ax.text(c, r, "FIRE", ha="center", va="center", fontsize=7, color="white", weight="bold")

        # Draw hostages on this floor
        for h_ent in env.hostages:
            if h_ent.floor == f and not h_ent.rescued:
                circle = plt.Circle(
                    (h_ent.col, h_ent.row),
                    0.35,
                    facecolor="#FFB300",
                    edgecolor="#E65100",
                    linewidth=1.5,
                    zorder=4,
                )
                ax.add_patch(circle)
                ax.text(
                    h_ent.col,
                    h_ent.row,
                    f"H{h_ent.id}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="#212121",
                    weight="bold",
                    zorder=5,
                )

        # Draw rescuers on this floor
        for ag_id, rescuer in env.rescuers.items():
            if rescuer.floor == f:
                r_color = "#2979FF" if rescuer.alive else "#9E9E9E"
                r_marker = "o" if rescuer.alive else "X"
                edge_col = "#FFD600" if rescuer.carrying else "#0D47A1"
                circle = plt.Circle(
                    (rescuer.col, rescuer.row),
                    0.4,
                    facecolor=r_color,
                    edgecolor=edge_col,
                    linewidth=2.0 if rescuer.carrying else 1.0,
                    zorder=6,
                )
                ax.add_patch(circle)
                tag = ag_id.replace("rescuer_", "R")
                if rescuer.carrying:
                    tag += "*"
                ax.text(
                    rescuer.col,
                    rescuer.row,
                    tag,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                    weight="bold",
                    zorder=7,
                )

        floor_name = "Ground Floor (Floor 0)" if f == 0 else f"Upper Floor (Floor {f})"
        ax.set_title(
            f"RADON Simulation — {floor_name} (Step {env.current_step})\n"
            f"[R*: carrying hostage | H: hostage | E: exit | S: stairs | F: fire]",
            fontsize=11,
            pad=10,
        )
        ax.set_xticks(range(0, w, 2))
        ax.set_yticks(range(0, h, 2))
        ax.grid(color="#CFD8DC", linestyle="--", linewidth=0.5, alpha=0.5)

        png_path = output_dir / f"day1_floor{f}.png"
        fig.tight_layout()
        fig.savefig(png_path)
        plt.close(fig)
        saved_paths.append(png_path)

    return saved_paths


def run_sanity_and_rollout():
    """Runs parallel_api_test, random agent rollout, replay export, and visualization."""
    print("=" * 70)
    print("  RADON Day 1: Environment & Plumbing Sanity Verification")
    print("=" * 70)

    # 1. PettingZoo ParallelEnv API Test
    print("\n[Step 1/4] Running PettingZoo ParallelEnv API test...")
    test_env = RadonRescueEnv(EnvConfig(seed=101))
    pettingzoo.test.parallel_api_test(test_env, num_cycles=250)
    print("  -> SUCCESS: PettingZoo parallel_api_test passed without errors.")

    # 2. Setup Env for Demo Rollout
    print("\n[Step 2/4] Initializing environment with fixed seed 42...")
    config = EnvConfig(seed=42)
    env = RadonRescueEnv(config=config)
    obs, infos = env.reset(seed=42)

    output_dir = Path("replays")
    output_dir.mkdir(parents=True, exist_ok=True)

    recorder = EpisodeRecorder(env)
    recorder.record_initial_frame()

    print(f"  Building generated with {env.config.floors} floors, {env.config.grid_h}x{env.config.grid_w} grid.")
    print(f"  Rescuers: {list(env.rescuers.keys())}")
    print(f"  Hostages: {len(env.hostages)} | Exits: {len(env.building_map.exits)} | Initial Fires: {len(env.building_map.fires)}")

    # 3. Execute Random Agent Rollout
    print("\n[Step 3/4] Running episode rollout with random policy...")
    total_rewards = {agent: 0.0 for agent in env.possible_agents}
    step_rewards_history = []

    while env.agents:
        current_step = env.current_step + 1
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, rew, terms, truncs, infos = env.step(actions)
        recorder.record_step(rew)

        step_total = sum(rew.values())
        step_rewards_history.append(step_total)
        for agent, r in rew.items():
            total_rewards[agent] += r

        rescued_count = sum(1 for h in env.hostages if h.rescued)
        living_count = sum(1 for r in env.rescuers.values() if r.alive)

        if current_step % 20 == 0 or not env.agents:
            print(
                f"  Step {current_step:03d} | Active Agents: {len(env.agents):1d} | "
                f"Living Rescuers: {living_count}/{len(env.possible_agents)} | "
                f"Rescued Hostages: {rescued_count}/{env.config.n_hostages} | "
                f"Step Reward Sum: {step_total:+6.2f}"
            )

    print("\n  Rollout Complete!")
    print(f"  Total Steps Elapsed: {env.current_step}")
    print(f"  Final Rescued Hostages: {sum(1 for h in env.hostages if h.rescued)}/{env.config.n_hostages}")
    print(f"  Living Rescuers: {sum(1 for r in env.rescuers.values() if r.alive)}/{len(env.possible_agents)}")
    for agent, r in total_rewards.items():
        print(f"    {agent}: Total Return = {r:+.2f}")

    # 4. Save Replay & Visualizations
    print("\n[Step 4/4] Saving artifacts...")
    replay_path = output_dir / "episode_demo.json"
    recorder.save(replay_path)
    print(f"  -> Saved replay JSON: {replay_path} ({len(recorder.frames)} frames)")

    saved_images = render_floors_to_png(env, output_dir)
    for p in saved_images:
        print(f"  -> Saved floor render: {p}")

    print("\nAll Day 1 checks passed successfully!\n")


if __name__ == "__main__":
    run_sanity_and_rollout()
