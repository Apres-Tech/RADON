"""Generates high-resolution learning curve plots from training metrics CSV with train/test generalization overlay."""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_learning_curves(
    csv_path: str | Path,
    output_path: str | Path = "metrics/day3_learning_curve.png",
    title: str | None = None,
):
    """Reads metrics CSV and generates publication-grade 2-panel learning curve plot with generalization breakdown."""
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"Metrics CSV not found at: {csv_file}")

    df = pd.read_csv(csv_file)
    timesteps = df["timestep"].values

    has_generalization = "train_success" in df.columns and "test_success" in df.columns
    has_survival = "mean_rescuers_surviving" in df.columns

    is_sanity = "sanity" in str(csv_path).lower()
    is_day3 = "day3" in str(csv_path).lower() or "floor" in str(csv_path).lower()
    is_day4_or_5 = any(k in str(csv_path).lower() for k in ["day4", "day5", "final", "adversary"])
    max_hostages = 1 if is_sanity else (2 if is_day4_or_5 else (4 if is_day3 else 3))
    max_rescuers = 2 if is_sanity else (3 if is_day4_or_5 else 4)

    num_panels = 3 if has_survival and has_generalization else 2
    fig_width = 20 if num_panels == 3 else 15
    fig, axes = plt.subplots(1, num_panels, figsize=(fig_width, 5.8), dpi=150)
    plt.subplots_adjust(wspace=0.28)

    ax1, ax2 = axes[0], axes[1]
    ax3 = axes[2] if num_panels == 3 else None

    # Style
    plt.rcParams["font.family"] = "sans-serif"
    grid_kwargs = dict(color="#E0E0E0", linestyle="--", linewidth=0.7, alpha=0.8)

    if has_generalization:
        train_rescued = df["train_rescued"].values
        test_rescued = df["test_rescued"].values
        train_success = df["train_success"].values * 100.0
        test_success = df["test_success"].values * 100.0

        # --- Panel 1: Success Rate (Train vs Held-out Test) ---
        ax1.plot(timesteps, train_success, color="#2E7D32", linewidth=2.2, marker="o", markersize=4, label="Train Success (%)")
        ax1.plot(timesteps, test_success, color="#E65100", linewidth=2.2, linestyle="--", marker="s", markersize=4, label="Held-Out Test Success (%)")
        ax1.fill_between(timesteps, test_success, train_success, color="#FFE0B2", alpha=0.35, label="Generalization Gap")
        ax1.set_xlabel("Environment Timesteps", fontsize=11, fontweight="bold")
        ax1.set_ylabel("All-Evacuated Success Rate (%)", fontsize=11, fontweight="bold")
        ax1.set_ylim(-5, 105)
        ax1.grid(True, **grid_kwargs)
        ax1.set_title("Success Rate: Train vs Held-Out Test", fontsize=12, fontweight="bold", pad=10)
        ax1.legend(loc="lower right", framealpha=0.9)

        # --- Panel 2: Hostages Rescued (Train vs Held-out Test) ---
        color_tr = "#1565C0"
        color_te = "#AD1457"
        ax2.plot(timesteps, train_rescued, color=color_tr, linewidth=2.2, marker="o", markersize=4, label=f"Train Rescued (/{max_hostages})")
        ax2.plot(timesteps, test_rescued, color=color_te, linewidth=2.2, linestyle="--", marker="s", markersize=4, label=f"Held-Out Test Rescued (/{max_hostages})")
        ax2.axhline(float(max_hostages), color="#90A4AE", linestyle=":", label=f"Max Capacity ({max_hostages})")
        ax2.set_xlabel("Environment Timesteps", fontsize=11, fontweight="bold")
        ax2.set_ylabel(f"Mean Hostages Rescued (out of {max_hostages})", fontsize=11, fontweight="bold")
        ax2.set_ylim(-0.05, max_hostages * 1.1)
        ax2.grid(True, **grid_kwargs)
        ax2.set_title("Hostages Evacuated: Train vs Held-Out Test", fontsize=12, fontweight="bold", pad=10)
        ax2.legend(loc="lower right", framealpha=0.9)

        # --- Panel 3: Survival Rate & Fire Avoidance ---
        if ax3 is not None:
            surviving = df["mean_rescuers_surviving"].values
            ax3.plot(timesteps, surviving, color="#00838F", linewidth=2.2, marker="^", markersize=4, label=f"Surviving Rescuers (/{max_rescuers})")
            ax3.axhline(float(max_rescuers), color="#80CBC4", linestyle=":", label=f"Team Size ({max_rescuers})")
            ax3.set_xlabel("Environment Timesteps", fontsize=11, fontweight="bold")
            ax3.set_ylabel(f"Surviving Rescuers (out of {max_rescuers})", color="#00838F", fontsize=11, fontweight="bold")
            ax3.set_ylim(-0.05, max_rescuers * 1.15)
            ax3.grid(True, **grid_kwargs)

            if "fire_deaths" in df.columns:
                fire_deaths = df["fire_deaths"].values
                ax3_twin = ax3.twinx()
                ax3_twin.plot(timesteps, fire_deaths, color="#C62828", linewidth=2.0, linestyle="--", marker="x", markersize=4, label="Fire Deaths")
                ax3_twin.set_ylabel("Fire Deaths", color="#C62828", fontsize=11, fontweight="bold")
                ax3_twin.set_ylim(-0.1, max(3.0, max(fire_deaths) * 1.2 if len(fire_deaths) else 3.0))

            ax3.set_title("Team Survival & Hazard Avoidance", fontsize=12, fontweight="bold", pad=10)
            ax3.legend(loc="lower left", framealpha=0.9)

    else:
        # Backward compatibility for legacy single-curve CSVs
        rescued = df["mean_hostages_rescued"].values
        success_rate = df["success_rate"].values * 100.0
        mean_reward = df["mean_reward"].values

        ax1.plot(timesteps, rescued, color="#1976D2", linewidth=2.2, marker="o", markersize=4, label=f"Rescued (/{max_hostages})")
        ax1.set_xlabel("Environment Timesteps", fontsize=11, fontweight="bold")
        ax1.set_ylabel(f"Hostages Rescued (out of {max_hostages})", color="#1976D2", fontsize=11, fontweight="bold")
        ax1.set_ylim(-0.05, max_hostages * 1.1)
        ax1.grid(True, **grid_kwargs)

        ax1_twin = ax1.twinx()
        ax1_twin.plot(timesteps, success_rate, color="#2E7D32", linewidth=2.0, linestyle="--", marker="s", markersize=3, label="Success Rate (%)")
        ax1_twin.set_ylabel("Success Rate (%)", color="#2E7D32", fontsize=11, fontweight="bold")
        ax1_twin.set_ylim(-5, 105)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", framealpha=0.9)

        ax2.plot(timesteps, mean_reward, color="#D84315", linewidth=2.2, marker="d", markersize=4, label="Mean Episode Return")
        ax2.set_xlabel("Environment Timesteps", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Mean Episode Return", color="#D84315", fontsize=11, fontweight="bold")
        ax2.grid(True, **grid_kwargs)
        ax2.set_title("Policy Return Convergence", fontsize=12, fontweight="bold", pad=10)
        ax2.legend(loc="lower right", framealpha=0.9)

    # Global Title
    if title is None:
        title = "RADON — Day 3 Multi-Agent Policy Generalization Curve (Train vs Held-Out Test)"
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Learning curve saved to: {out_file}")
    return out_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot RADON Learning Curves")
    parser.add_argument("--csv-path", type=str, required=True, help="Path to training metrics CSV")
    parser.add_argument("--output-path", type=str, default="metrics/day3_learning_curve.png", help="Output image path")
    parser.add_argument("--title", type=str, default=None, help="Custom figure title")
    args = parser.parse_args()
    plot_learning_curves(args.csv_path, args.output_path, args.title)
