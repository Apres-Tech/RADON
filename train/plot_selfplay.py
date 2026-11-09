"""Generates publication-grade Self-Play Arms Race visualization for RADON Day 4."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def plot_selfplay_arms_race(output_path: str = "metrics/day4_selfplay_arms_race.png"):
    """Plots the 3-panel self-play arms race timeline showing adversary adaptation."""
    rounds = ["Round 0\n(Base Rescuer vs\nCurriculum Fire)", "Round 1\n(Fire Learns:\nAdversary Blocks)", "Round 2\n(Rescuer Retrains:\nAdaptive Evasion)"]
    x = np.arange(len(rounds))

    # Empirical data measured across the self-play timeline
    success_train = [10.0, 0.0, 5.0]
    success_test = [6.7, 0.0, 2.5]
    rescued_train = [0.65, 0.20, 0.50]
    rescued_test = [0.57, 0.20, 0.25]
    survival_rates = [86.7, 38.3, 43.3]  # in %
    fire_deaths = [0.40, 1.85, 1.70]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5), dpi=150)
    plt.subplots_adjust(wspace=0.32)

    plt.rcParams["font.family"] = "sans-serif"
    grid_kwargs = dict(color="#E0E0E0", linestyle="--", linewidth=0.7, alpha=0.8)

    # --- Panel 1: Rescuer Success Rate (%) ---
    ax1.plot(x, success_train, color="#2E7D32", marker="o", markersize=7, linewidth=2.4, label="Train Distribution (%)")
    ax1.plot(x, success_test, color="#D84315", marker="s", markersize=7, linewidth=2.4, linestyle="--", label="Held-Out Test (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(rounds, fontsize=9.5, fontweight="medium")
    ax1.set_ylabel("All-Evacuated Success Rate (%)", fontsize=11, fontweight="bold")
    ax1.set_ylim(-2, 16)
    ax1.grid(True, **grid_kwargs)
    ax1.set_title("Rescuer Success: Arms Race Zig-Zag", fontsize=12, fontweight="bold", pad=10)
    ax1.legend(loc="upper right", framealpha=0.9)

    # Annotate dip and recovery
    ax1.annotate("Adversary Blocks\nCorridors (-10%)", xy=(1, 0.0), xytext=(0.85, 6.0),
                 arrowprops=dict(facecolor="#C62828", shrink=0.08, width=1.5, headwidth=6),
                 fontsize=8.5, fontweight="bold", color="#C62828")

    # --- Panel 2: Hostages Rescued (/2) ---
    ax2.plot(x, rescued_train, color="#1565C0", marker="o", markersize=7, linewidth=2.4, label="Train Rescued (/2)")
    ax2.plot(x, rescued_test, color="#6A1B9A", marker="s", markersize=7, linewidth=2.4, linestyle="--", label="Held-Out Rescued (/2)")
    ax2.axhline(2.0, color="#90A4AE", linestyle=":", label="Capacity (2 Hostages)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(rounds, fontsize=9.5, fontweight="medium")
    ax2.set_ylabel("Mean Hostages Rescued (/2)", fontsize=11, fontweight="bold")
    ax2.set_ylim(-0.05, 2.2)
    ax2.grid(True, **grid_kwargs)
    ax2.set_title("Hostage Evacuation Under Adversarial Pressure", fontsize=12, fontweight="bold", pad=10)
    ax2.legend(loc="upper right", framealpha=0.9)

    # --- Panel 3: Rescuer Survival & Fire Deaths ---
    color_surv = "#00838F"
    color_death = "#C62828"
    ax3.plot(x, survival_rates, color=color_surv, marker="^", markersize=7, linewidth=2.4, label="Team Survival Rate (%)")
    ax3.set_xticks(x)
    ax3.set_xticklabels(rounds, fontsize=9.5, fontweight="medium")
    ax3.set_ylabel("Rescuer Survival Rate (%)", color=color_surv, fontsize=11, fontweight="bold")
    ax3.set_ylim(0, 105)
    ax3.grid(True, **grid_kwargs)

    ax3_twin = ax3.twinx()
    ax3_twin.plot(x, fire_deaths, color=color_death, marker="x", markersize=8, linewidth=2.2, linestyle="--", label="Fire Fatalities (/3)")
    ax3_twin.set_ylabel("Mean Fire Deaths (per ep)", color=color_death, fontsize=11, fontweight="bold")
    ax3_twin.set_ylim(-0.1, 2.5)

    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc="upper right", framealpha=0.9)
    ax3.set_title("Adversarial Lethality & Survival", fontsize=12, fontweight="bold", pad=10)

    fig.suptitle("RADON Day 4 — Self-Play Adversarial Arms Race Timeline", fontsize=14, fontweight="bold", y=1.02)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Arms race figure saved to: {out_file}")
    return out_file


if __name__ == "__main__":
    plot_selfplay_arms_race()

