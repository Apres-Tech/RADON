"""Day 10 Iterated Best-Response Self-Play Training on Large Maze Maps.

Arms-race rounds:
- Round 0: Train Rescuer on maze maps against Curriculum Fire (learning navigation in mazes + HP/ash mechanics).
- Round 1: Freeze Rescuer R0; train Fire Adversary PPO (learning chokepoint ambushes against strong rescuers).
- Round 2: Freeze Fire R1; retrain Rescuer R2 (learning to adapt to fire chokepoint pressure, wait for ash, dash with i-frames).
- Log per-round metrics & generate self-play arms-race curves.
"""

import argparse
import csv
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from env.config import EnvConfig
from env.fire_controller import CurriculumFireController, PolicyFireController
from env.fire_env import FireAdversaryEnv
from env.rescue_env import RadonRescueEnv, Tile
from train.extractor import RescueExtractor
from train.make_env import make_rescue_vec_env


class FireFeatureExtractor(BaseFeaturesExtractor):
    """CNN + MLP feature extractor for the Fire adversary's 10x10 canonical multi-floor observation."""

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 160):
        super().__init__(observation_space, features_dim)
        grid_shape = observation_space.spaces["grid"].shape
        scalar_dim = observation_space.spaces["scalars"].shape[0]

        in_channels = grid_shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy_grid = torch.as_tensor(observation_space.spaces["grid"].sample()[None]).float()
            cnn_out_dim = self.cnn(dummy_grid).shape[1]

        self.grid_proj = nn.Sequential(
            nn.Linear(cnn_out_dim, 128),
            nn.ReLU(),
        )

        self.scalar_proj = nn.Sequential(
            nn.Linear(scalar_dim, 32),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(128 + 32, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict) -> torch.Tensor:
        grid_feats = self.grid_proj(self.cnn(observations["grid"]))
        scalar_feats = self.scalar_proj(observations["scalars"])
        combined = torch.cat([grid_feats, scalar_feats], dim=1)
        return self.fc(combined)


def evaluate_matchup(
    rescuer_model: PPO,
    fire_controller,
    config: EnvConfig,
    num_episodes: int = 25,
    base_seed: int = 50000,
) -> Dict[str, float]:
    """Evaluates a rescuer policy against a fire controller on held-out seeds."""
    full_wins = 0
    partial_wins = 0
    rescued_list = []
    survivor_list = []
    deaths_list = []
    hp_lost_list = []
    dashes_list = []
    length_list = []

    for ep_idx in range(num_episodes):
        ep_seed = base_seed + ep_idx
        env = RadonRescueEnv(config=config, fire_controller=fire_controller)
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
        survivor_list.append(survivors)
        deaths_list.append(getattr(env, "fire_deaths", config.n_rescuers - survivors))
        hp_lost_list.append(getattr(env, "hp_lost", 0))
        dashes_list.append(getattr(env, "fire_dashes_survived", 0))
        length_list.append(env.current_step)

    return {
        "full_win_rate": round(full_wins / num_episodes * 100.0, 1),
        "partial_win_rate": round(partial_wins / num_episodes * 100.0, 1),
        "mean_rescued": round(float(np.mean(rescued_list)), 2),
        "rescued_pct": round(float(np.mean(rescued_list)) / config.n_hostages * 100.0, 1),
        "mean_survivors": round(float(np.mean(survivor_list)), 2),
        "mean_deaths": round(float(np.mean(deaths_list)), 2),
        "mean_hp_lost": round(float(np.mean(hp_lost_list)), 2),
        "mean_dashes_survived": round(float(np.mean(dashes_list)), 2),
        "mean_length": round(float(np.mean(length_list)), 1),
    }


def plot_day10_arms_race(csv_path: str, output_path: str = "metrics/day10_selfplay_arms_race.png") -> None:
    """Generates a 3-panel visualization of the Day 10 self-play arms race."""
    rounds = []
    full_wins = []
    mean_rescued = []
    fire_deaths = []
    survivors = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rounds.append(row["round_label"])
            full_wins.append(float(row["full_win_rate"]))
            mean_rescued.append(float(row["mean_rescued"]))
            fire_deaths.append(float(row["mean_deaths"]))
            survivors.append(float(row["mean_survivors"]))

    x = np.arange(len(rounds))
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.2), dpi=150)
    plt.subplots_adjust(wspace=0.30)
    grid_kw = dict(color="#E0E0E0", linestyle="--", linewidth=0.7, alpha=0.8)

    # Panel 1: Full-Win Rate
    ax1.plot(x, full_wins, color="#16a34a", marker="o", markersize=8, linewidth=2.5, label="Held-Out Win Rate (%)")
    ax1.axhline(50.0, color="#f59e0b", linestyle=":", linewidth=1.8, label="Target Fair Fight (50%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(rounds, fontsize=9.5)
    ax1.set_ylabel("Full Win Rate (All Evacuated %)", fontsize=11, fontweight="bold")
    ax1.set_ylim(-5, 105)
    ax1.grid(True, **grid_kw)
    ax1.set_title("Rescuer Win Rate Across Self-Play", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right", framealpha=0.9)

    # Panel 2: Mean Hostages Rescued
    ax2.plot(x, mean_rescued, color="#2563eb", marker="s", markersize=8, linewidth=2.5, label="Hostages Rescued")
    ax2.axhline(4.0, color="#94a3b8", linestyle=":", linewidth=1.5, label="Max Hostages (4)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(rounds, fontsize=9.5)
    ax2.set_ylabel("Mean Rescued per Episode", fontsize=11, fontweight="bold")
    ax2.set_ylim(-0.2, 4.4)
    ax2.grid(True, **grid_kw)
    ax2.set_title("Hostage Evacuation Volume", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper right", framealpha=0.9)

    # Panel 3: Survivability & Lethality
    ax3.plot(x, survivors, color="#0d9488", marker="^", markersize=8, linewidth=2.5, label="Surviving Rescuers (/5)")
    ax3_twin = ax3.twinx()
    ax3_twin.plot(x, fire_deaths, color="#dc2626", marker="x", markersize=8, linewidth=2.2, linestyle="--", label="Fire Fatalities")
    ax3.set_xticks(x)
    ax3.set_xticklabels(rounds, fontsize=9.5)
    ax3.set_ylabel("Living Rescuers", color="#0d9488", fontsize=11, fontweight="bold")
    ax3_twin.set_ylabel("Fire Deaths", color="#dc2626", fontsize=11, fontweight="bold")
    ax3.set_ylim(0, 5.5)
    ax3_twin.set_ylim(-0.2, 3.5)
    ax3.grid(True, **grid_kw)
    ax3.set_title("Team Survival vs Hazard Mortality", fontsize=12, fontweight="bold")

    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc="upper right", framealpha=0.9)

    fig.suptitle("RADON Day 10 — Self-Play Adversarial Arms Race on Large Maze Maps", fontsize=14, fontweight="bold", y=1.02)
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_p, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved arms race curve to {out_p}")


def run_selfplay_rounds(
    rescuer_steps_r0: int = 50000,
    fire_steps_r1: int = 35000,
    rescuer_steps_r2: int = 50000,
    fire_steps_r2: int = 30000,
    num_eval_episodes: int = 25,
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Starting Day 10 Iterated Best-Response Self-Play (Device: {device}) ===")

    agents_dir = Path("agents")
    agents_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = Path("metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    maze_cfg = EnvConfig.maze(grid_size=20)

    # -------------------------------------------------------------
    # ROUND 0: Train Rescuer R0 against Curriculum Fire
    # -------------------------------------------------------------
    r0_path = agents_dir / "day10_rescuer_r0.zip"
    if r0_path.exists() and rescuer_steps_r0 == 0:
        print(f"\n[ROUND 0] Reusing existing Rescuer R0 checkpoint: {r0_path}")
        rescuer_r0 = PPO.load(str(r0_path), device=device)
    else:
        print("\n[ROUND 0] Training Rescuer R0 on Maze maps with Curriculum Fire...")
        vec_env_r0 = make_rescue_vec_env(
            config=maze_cfg,
            num_vec_envs=6,
            num_cpus=1,
            fire_controller_factory=lambda: CurriculumFireController(),
        )

        policy_kwargs_rescuer = dict(
            features_extractor_class=RescueExtractor,
            features_extractor_kwargs=dict(features_dim=160),
            net_arch=dict(pi=[128, 64], vf=[128, 64]),
        )

        rescuer_r0 = PPO(
            "MultiInputPolicy",
            vec_env_r0,
            policy_kwargs=policy_kwargs_rescuer,
            n_steps=512,
            batch_size=128,
            learning_rate=3e-4,
            ent_coef=0.01,
            gamma=0.99,
            gae_lambda=0.95,
            verbose=0,
            device=device,
        )
        t0 = time.time()
        rescuer_r0.learn(total_timesteps=rescuer_steps_r0)
        print(f"[ROUND 0] Rescuer R0 trained in {time.time() - t0:.1f}s. Saving model...")
        rescuer_r0.save(str(r0_path))
        vec_env_r0.close()

    # Evaluate R0 vs Curriculum Fire
    r0_curric_eval = evaluate_matchup(
        rescuer_model=rescuer_r0,
        fire_controller=CurriculumFireController(),
        config=maze_cfg,
        num_episodes=num_eval_episodes,
    )
    print(f"[EVAL] Round 0 (Rescuer R0 vs Curriculum Fire): Win Rate = {r0_curric_eval['full_win_rate']}%, Rescued = {r0_curric_eval['mean_rescued']}/4")

    # -------------------------------------------------------------
    # ROUND 1: Train Fire Adversary R1 against Frozen Rescuer R0
    # -------------------------------------------------------------
    print("\n[ROUND 1] Training Fire Adversary R1 against Frozen Rescuer R0...")
    fire_env_r1 = FireAdversaryEnv(
        config=maze_cfg,
        rescuer_policy_path=str(r0_path),
        deterministic_rescuers=False,
    )

    policy_kwargs_fire = dict(
        features_extractor_class=FireFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=160),
        net_arch=dict(pi=[128, 64], vf=[128, 64]),
    )

    fire_r1 = PPO(
        "MultiInputPolicy",
        fire_env_r1,
        policy_kwargs=policy_kwargs_fire,
        n_steps=256,
        batch_size=64,
        learning_rate=4e-4,
        ent_coef=0.02,
        gamma=0.98,
        verbose=0,
        device=device,
    )
    t0 = time.time()
    fire_r1.learn(total_timesteps=fire_steps_r1)
    print(f"[ROUND 1] Fire Adversary R1 trained in {time.time() - t0:.1f}s. Saving model...")
    f1_path = agents_dir / "day10_fire_r1.zip"
    fire_r1.save(str(f1_path))
    fire_env_r1.close()

    # Evaluate R0 vs Trained Fire R1 (Expect sharp drop in rescuer win rate)
    f1_controller = PolicyFireController(model=fire_r1, deterministic=False)
    r0_vs_f1_eval = evaluate_matchup(
        rescuer_model=rescuer_r0,
        fire_controller=f1_controller,
        config=maze_cfg,
        num_episodes=num_eval_episodes,
    )
    print(f"[EVAL] Round 1 (Rescuer R0 vs Fire Adversary R1): Win Rate = {r0_vs_f1_eval['full_win_rate']}%, Rescued = {r0_vs_f1_eval['mean_rescued']}/4")

    # -------------------------------------------------------------
    # ROUND 2: Retrain Rescuer R2 against Frozen Fire Adversary R1
    # -------------------------------------------------------------
    print("\n[ROUND 2] Retraining Rescuer R2 against Frozen Fire Adversary R1...")
    vec_env_r2 = make_rescue_vec_env(
        config=maze_cfg,
        num_vec_envs=6,
        num_cpus=1,
        fire_controller_factory=lambda: PolicyFireController(model=fire_r1, deterministic=False),
    )

    # Resume rescuer learning from R0 checkpoint
    rescuer_r2 = PPO.load(str(r0_path), env=vec_env_r2, device=device)
    t0 = time.time()
    rescuer_r2.learn(total_timesteps=rescuer_steps_r2)
    print(f"[ROUND 2] Rescuer R2 retrained in {time.time() - t0:.1f}s. Saving model...")
    r2_path = agents_dir / "day10_rescuer_r2.zip"
    rescuer_r2.save(str(r2_path))
    vec_env_r2.close()

    # Evaluate R2 vs Fire R1 (Expect recovery!)
    r2_vs_f1_eval = evaluate_matchup(
        rescuer_model=rescuer_r2,
        fire_controller=f1_controller,
        config=maze_cfg,
        num_episodes=num_eval_episodes,
    )
    print(f"[EVAL] Round 2 (Rescuer R2 vs Fire Adversary R1): Win Rate = {r2_vs_f1_eval['full_win_rate']}%, Rescued = {r2_vs_f1_eval['mean_rescued']}/4")

    # -------------------------------------------------------------
    # ROUND 3: Fine-tune Fire Adversary R2 against Rescuer R2 (Arms-Race Balance)
    # -------------------------------------------------------------
    print("\n[ROUND 3] Fine-tuning Fire Adversary R2 against Rescuer R2...")
    fire_env_r2 = FireAdversaryEnv(
        config=maze_cfg,
        rescuer_policy_path=str(r2_path),
        deterministic_rescuers=False,
    )
    fire_r2 = PPO.load(str(f1_path), env=fire_env_r2, device=device)
    t0 = time.time()
    fire_r2.learn(total_timesteps=fire_steps_r2)
    print(f"[ROUND 3] Fire Adversary R2 tuned in {time.time() - t0:.1f}s. Saving model...")
    f2_path = agents_dir / "day10_fire_r2.zip"
    fire_r2.save(str(f2_path))
    fire_env_r2.close()

    # Evaluate R2 vs Fire R2 (Balanced Fight)
    f2_controller = PolicyFireController(model=fire_r2, deterministic=False)
    r2_vs_f2_eval = evaluate_matchup(
        rescuer_model=rescuer_r2,
        fire_controller=f2_controller,
        config=maze_cfg,
        num_episodes=num_eval_episodes,
    )
    print(f"[EVAL] Round 3 (Rescuer R2 vs Fire Adversary R2): Win Rate = {r2_vs_f2_eval['full_win_rate']}%, Rescued = {r2_vs_f2_eval['mean_rescued']}/4")

    # Record arms race history to CSV
    csv_file = metrics_dir / "day10_selfplay_arms_race.csv"
    rounds_data = [
        {"round": 0, "round_label": "R0: Rescuer vs\nCurriculum Fire", "matchup": "Rescuer R0 vs Curriculum", **r0_curric_eval},
        {"round": 1, "round_label": "R1: Fire Learns\nChokepoints", "matchup": "Rescuer R0 vs Fire R1", **r0_vs_f1_eval},
        {"round": 2, "round_label": "R2: Rescuer Adapts\n(Ash & Evasion)", "matchup": "Rescuer R2 vs Fire R1", **r2_vs_f1_eval},
        {"round": 3, "round_label": "R3: Balanced Fight\n(Both Retrained)", "matchup": "Rescuer R2 vs Fire R2", **r2_vs_f2_eval},
    ]

    fieldnames = list(rounds_data[0].keys())
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rounds_data:
            writer.writerow(r)
    print(f"[CSV] Wrote arms race metrics to {csv_file}")

    # Plot arms race visualization
    plot_day10_arms_race(str(csv_file), str(metrics_dir / "day10_selfplay_arms_race.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--r0-steps", type=int, default=50000)
    parser.add_argument("--f1-steps", type=int, default=30000)
    parser.add_argument("--r2-steps", type=int, default=45000)
    parser.add_argument("--f2-steps", type=int, default=25000)
    parser.add_argument("--eval-episodes", type=int, default=25)
    args = parser.parse_args()

    run_selfplay_rounds(
        rescuer_steps_r0=args.r0_steps,
        fire_steps_r1=args.f1_steps,
        rescuer_steps_r2=args.r2_steps,
        fire_steps_r2=args.f2_steps,
        num_eval_episodes=args.eval_episodes,
    )

