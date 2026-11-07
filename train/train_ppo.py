"""PPO training entrypoint for RADON multi-agent rescue environment."""

import argparse
from pathlib import Path
import time
import torch
from stable_baselines3 import PPO

from env.config import EnvConfig
from train.callbacks import MetricsCallback
from train.extractor import RescueExtractor
from train.make_env import make_rescue_vec_env


def parse_args():
    """Parses command-line training arguments."""
    parser = argparse.ArgumentParser(description="RADON Day 2 PPO Training Pipeline")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=300_000,
        help="Total training timesteps (default: 300,000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for environment and policy (default: 42)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="run1",
        help="Experiment identifier name (default: 'run1')",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=512,
        help="Number of steps per environment per PPO rollout (default: 512)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="PPO minibatch size (default: 256)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="PPO learning rate (default: 3e-4)",
    )
    parser.add_argument(
        "--num-vec-envs",
        type=int,
        default=2,
        help="Number of parallel building env copies (default: 2 -> 8 agent envs, <= 6)",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=10000,
        help="Frequency in timesteps for running evaluation and logging (default: 10000)",
    )
    return parser.parse_args()


def train(args):
    """Initializes and runs PPO training."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70)
    print("  RADON Day 2: PPO Multi-Agent Rescue Training")
    print("=" * 70)
    print(f"  Run Name:       {args.run_name}")
    print(f"  Target Steps:   {args.timesteps:,}")
    print(f"  Compute Device: {device.upper()} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")
    print(f"  Seed:           {args.seed}")
    print(f"  Vec Envs:       {args.num_vec_envs} building copies ({args.num_vec_envs * 4} agents)")
    print(f"  PPO Hparams:    n_steps={args.n_steps}, batch_size={args.batch_size}, lr={args.lr}")
    print("=" * 70)

    # 1. Environment creation
    config = EnvConfig.training_day2(seed=args.seed)
    vec_env = make_rescue_vec_env(
        config=config,
        num_vec_envs=args.num_vec_envs,
        num_cpus=1,
        seed=args.seed,
    )

    # 2. Policy configuration
    policy_kwargs = dict(
        features_extractor_class=RescueExtractor,
        features_extractor_kwargs=dict(features_dim=160),
        net_arch=dict(pi=[128, 64], vf=[128, 64]),
    )

    model = PPO(
        policy="MultiInputPolicy",
        env=vec_env,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        device=device,
        verbose=0,
        seed=args.seed,
    )

    # 3. Setup Metrics Callback
    metrics_callback = MetricsCallback(
        eval_freq=args.eval_freq,
        eval_episodes=5,
        config=config,
        log_dir="metrics",
        run_name=args.run_name,
        verbose=1,
    )

    agents_dir = Path("agents")
    agents_dir.mkdir(parents=True, exist_ok=True)

    print("\nStarting PPO rollout and optimization loop...")
    start_time = time.time()

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=metrics_callback,
            reset_num_timesteps=True,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current checkpoint...")

    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time/60:.2f} minutes ({elapsed_time:.1f}s).")

    # 4. Save final model checkpoint
    save_path = agents_dir / f"ppo_day2_{args.run_name}.zip"
    model.save(save_path)
    print(f"Model saved to: {save_path}")

    # Final post-training evaluation
    print("\n[Post-Training Final Evaluation]")
    final_metrics = metrics_callback._evaluate_policy(model=model, num_episodes=20)
    print(
        f"  Final Success Rate:    {final_metrics['success_rate']*100:5.1f}%\n"
        f"  Mean Hostages Rescued: {final_metrics['mean_hostages_rescued']:4.2f}/3\n"
        f"  Mean Episode Return:   {final_metrics['mean_reward']:+7.2f}\n"
        f"  Mean Surviving Agents: {final_metrics['mean_rescuers_surviving']:3.1f}/4"
    )

    vec_env.close()
    return model, save_path


if __name__ == "__main__":
    args = parse_args()
    train(args)
