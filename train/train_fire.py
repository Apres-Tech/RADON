"""Training script for the Adversarial Fire Policy in RADON self-play."""

import argparse
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv

from env.config import EnvConfig
from env.fire_env import FireAdversaryEnv
from env.rescue_env import RadonRescueEnv
from env.fire_controller import PolicyFireController, CurriculumFireController
from train.eval_protocol import TEST_EVAL_SEEDS


class FireFeatureExtractor(BaseFeaturesExtractor):
    """Custom CNN + MLP extractor for the Fire adversary's multi-floor global observation."""

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


def evaluate_rescuer_vs_fire(
    rescuer_model_path: str,
    fire_controller,
    config: EnvConfig,
    num_episodes: int = 20,
    seed_start: int = 50000,
) -> dict:
    """Evaluates rescuer team performance against a specific fire controller on held-out seeds."""
    from stable_baselines3 import PPO
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rescuer_model = PPO.load(rescuer_model_path, device=device)

    env = RadonRescueEnv(config=config, fire_controller=fire_controller)
    successes = []
    rescued_counts = []
    surviving_counts = []
    fire_deaths = []

    for ep_idx in range(num_episodes):
        seed = seed_start + ep_idx
        obs, info = env.reset(seed=seed)

        while env.agents:
            actions = {}
            for aid in env.agents:
                a_obs = {k: np.expand_dims(v, axis=0) for k, v in obs[aid].items()}
                act, _ = rescuer_model.predict(a_obs, deterministic=False)
                actions[aid] = int(act[0])
            obs, rew, term, trunc, info = env.step(actions)

        rescued = sum(1 for h in env.hostages if h.rescued)
        successes.append(1.0 if rescued == config.n_hostages else 0.0)
        rescued_counts.append(rescued)
        surviving_counts.append(sum(1 for r in env.rescuers.values() if r.alive))
        fire_deaths.append(env.fire_deaths)

    env.close()
    return {
        "success_rate": float(np.mean(successes)),
        "mean_rescued": float(np.mean(rescued_counts)),
        "mean_surviving": float(np.mean(surviving_counts)),
        "mean_fire_deaths": float(np.mean(fire_deaths)),
    }


def train_fire(args):
    """Trains the fire adversary policy against a frozen rescuer baseline."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = EnvConfig.training_day4_base()

    print("=" * 70)
    print("  RADON Self-Play: Training Fire Adversary (Round 1)")
    print(f"  Frozen Rescuer: {args.rescuer_model}")
    print(f"  Target Steps:   {args.timesteps:,}")
    print(f"  Device:         {device.upper()}")
    print("=" * 70)

    # 1. Baseline Evaluation: Rescuers vs Curriculum Fire (Round 0)
    print("\n[Self-Play Round 0 Baseline] Evaluating Frozen Rescuer vs Curriculum Fire...")
    r0_eval = evaluate_rescuer_vs_fire(
        rescuer_model_path=args.rescuer_model,
        fire_controller=CurriculumFireController(),
        config=config,
        num_episodes=20,
    )
    print(
        f"  R0 Baseline -> Rescuer Success: {r0_eval['success_rate']*100:.1f}% | "
        f"Rescued: {r0_eval['mean_rescued']:.2f}/2 | "
        f"Surviving: {r0_eval['mean_surviving']:.2f}/3 | "
        f"Fire Deaths: {r0_eval['mean_fire_deaths']:.2f}"
    )

    # 2. Setup Fire Training Environment
    def make_fire_env():
        return FireAdversaryEnv(
            config=config,
            rescuer_policy_path=args.rescuer_model,
            deterministic_rescuers=False,
        )

    vec_env = make_vec_env(make_fire_env, n_envs=args.num_envs, vec_env_cls=DummyVecEnv)

    policy_kwargs = dict(
        features_extractor_class=FireFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=160),
        net_arch=dict(pi=[128, 64], vf=[128, 64]),
    )

    fire_model = PPO(
        policy="MultiInputPolicy",
        env=vec_env,
        learning_rate=args.lr,
        n_steps=256,
        batch_size=64,
        n_epochs=6,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.03,
        vf_coef=0.5,
        policy_kwargs=policy_kwargs,
        device=device,
        verbose=1,
        seed=args.seed,
    )

    print("\nTraining Fire Adversary Policy...")
    start_t = time.time()
    fire_model.learn(total_timesteps=args.timesteps)
    elapsed = time.time() - start_t
    print(f"\nFire training completed in {elapsed/60:.2f} minutes.")

    # 3. Save Fire Model
    out_path = Path(args.output_model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fire_model.save(str(out_path))
    print(f"Saved trained fire model to: {out_path}")

    # 4. Round 1 Evaluation: Rescuers vs Trained Fire Adversary
    print("\n[Self-Play Round 1 Evaluation] Evaluating Frozen Rescuer vs Trained Fire (R1)...")
    fire_ctrl = PolicyFireController(model_path=str(out_path))
    r1_eval = evaluate_rescuer_vs_fire(
        rescuer_model_path=args.rescuer_model,
        fire_controller=fire_ctrl,
        config=config,
        num_episodes=20,
    )
    print(
        f"  R1 Adversary -> Rescuer Success: {r1_eval['success_rate']*100:.1f}% | "
        f"Rescued: {r1_eval['mean_rescued']:.2f}/2 | "
        f"Surviving: {r1_eval['mean_surviving']:.2f}/3 | "
        f"Fire Deaths: {r1_eval['mean_fire_deaths']:.2f}"
    )

    diff_succ = r1_eval["success_rate"] - r0_eval["success_rate"]
    diff_resc = r1_eval["mean_rescued"] - r0_eval["mean_rescued"]
    print("\n" + "=" * 70)
    print("  SELF-PLAY ROUND 1 RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Rescuer Success: {r0_eval['success_rate']*100:.1f}% (R0) -> {r1_eval['success_rate']*100:.1f}% (R1) (Delta: {diff_succ*100:+.1f}%)")
    print(f"  Mean Rescued:    {r0_eval['mean_rescued']:.2f}/2 (R0) -> {r1_eval['mean_rescued']:.2f}/2 (R1) (Delta: {diff_resc:+.2f})")
    print(f"  Fire Deaths:     {r0_eval['mean_fire_deaths']:.2f} (R0) -> {r1_eval['mean_fire_deaths']:.2f} (R1)")
    print("=" * 70)

    return r0_eval, r1_eval


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RADON Fire Adversary Policy")
    parser.add_argument("--rescuer-model", type=str, default="agents/base_frozen.zip", help="Path to frozen rescuer model")
    parser.add_argument("--output-model", type=str, default="agents/fire_r1.zip", help="Path to save trained fire model")
    parser.add_argument("--timesteps", type=int, default=30_000, help="Timesteps to train fire")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--num-envs", type=int, default=2, help="Number of parallel environments")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    train_fire(args)

