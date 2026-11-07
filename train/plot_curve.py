"""Generates high-resolution learning curve plots from training metrics CSV."""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_learning_curves(
    csv_path: str | Path,
    output_path: str | Path = "metrics/day2_learning_curve.png",
    title: str | None = None,
):
    """Reads metrics CSV and generates publication-grade 2-panel learning curve plot."""
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"Metrics CSV not found at: {csv_file}")

    df = pd.read_csv(csv_file)
    timesteps = df["timestep"].values
    rescued = df["mean_hostages_rescued"].values
    success_rate = df["success_rate"].values * 100.0  # Percentage
    mean_reward = df["mean_reward"].values

    is_sanity = "sanity" in str(csv_path).lower()
    max_hostages = 1 if is_sanity else 3

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)
    plt.subplots_adjust(wspace=0.25)

    # Style
    plt.rcParams["font.family"] = "sans-serif"
    grid_kwargs = dict(color="#E0E0E0", linestyle="--", linewidth=0.7, alpha=0.8)

    # --- Panel 1: Success Rate & Hostages Rescued ---
    color_hostages = "#1976D2"  # Deep Blue
    color_success = "#2E7D32"   # Forest Green

    ax1.plot(timesteps, rescued, color=color_hostages, linewidth=2.2, marker="o", markersize=4, label=f"Mean Hostages Rescued (/{max_hostages})")
    ax1.set_xlabel("Environment Timesteps", fontsize=11, fontweight="bold")
    ax1.set_ylabel(f"Hostages Rescued (out of {max_hostages})", color=color_hostages, fontsize=11, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=color_hostages)
    ax1.set_ylim(-0.05, max_hostages * 1.1)
    ax1.axhline(float(max_hostages), color="#B0BEC5", linestyle=":", label=f"Max Capacity ({max_hostages} Hostage{'s' if max_hostages > 1 else ''})")
    ax1.grid(True, **grid_kwargs)

    ax1_twin = ax1.twinx()
    ax1_twin.plot(timesteps, success_rate, color=color_success, linewidth=2.0, linestyle="--", marker="s", markersize=3, label="Success Rate (%)")
    ax1_twin.set_ylabel("All-Evacuated Success Rate (%)", color=color_success, fontsize=11, fontweight="bold")
    ax1_twin.tick_params(axis="y", labelcolor=color_success)
    ax1_twin.set_ylim(-5, 105)

    ax1.set_title("Evacuation Performance vs Timesteps", fontsize=12, fontweight="bold", pad=10)

    # Combine legends for ax1 and ax1_twin
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", framealpha=0.9)

    # --- Panel 2: Mean Episode Return ---
    color_reward = "#D84315"  # Burnt Orange
    ax2.plot(timesteps, mean_reward, color=color_reward, linewidth=2.2, marker="d", markersize=4, label="Mean Episode Return")
    ax2.set_xlabel("Environment Timesteps", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Mean Episode Return", color=color_reward, fontsize=11, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=color_reward)
    ax2.grid(True, **grid_kwargs)
    ax2.set_title("Policy Return Convergence", fontsize=12, fontweight="bold", pad=10)
    ax2.legend(loc="lower right", framealpha=0.9)

    # Global Title
    if title is None:
        if is_sanity:
            title = (
                "RADON — Day 2.5 Sanity Curriculum Verification\n"
                "(Single Floor 8x8, 2 Rescuers, 1 Hostage, No Fire)"
            )
        else:
            title = (
                "RADON — Day 2.5 PPO Hardened Rescue Policy Learning Curve\n"
                "(Single Floor 12x12, 4 Shared Rescuers, 3 Hostages, BFS Shaping)"
            )
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Learning curve saved to: {out_file}")
    return out_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot RADON Day 2 Learning Curves")
    parser.add_argument(
        "--csv-path",
        type=str,
        default="metrics/day2_rescue_team.csv",
        help="Path to training metrics CSV",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="metrics/day2_learning_curve.png",
        help="Output image path",
    )
    args = parser.parse_args()
    plot_learning_curves(args.csv_path, args.output_path)

