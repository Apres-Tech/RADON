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
        default=250_000,
        help="Total training timesteps (default: 250,000)",
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=["training_day4_base", "training_day3", "training_day2", "sanity"],
        default="training_day4_base",
        help="Configuration preset: 'training_day4_base', 'training_day3', 'training_day2', or 'sanity' (default: 'training_day4_base')",
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
        default="rescue_team",
        help="Experiment identifier name (default: 'rescue_team')",
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
        default=128,
        help="PPO minibatch size (default: 128)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=4e-4,
        help="PPO learning rate (default: 4e-4)",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.04,
        help="PPO entropy coefficient for action exploration (default: 0.04)",
    )
    parser.add_argument(
        "--num-vec-envs",
        type=int,
        default=2,
        help="Number of parallel building env copies (default: 2)",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=10000,
        help="Frequency in timesteps for running evaluation and logging (default: 10000)",
    )
    parser.add_argument(
        "--fixed-map",
        action="store_true",
        help="Freeze map generation to a single static seed (default: False, fresh maps every episode)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to an existing model zip file to resume training from (default: None)",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes per distribution (train/test) per eval checkpoint (default: 10)",
    )
    parser.add_argument(
        "--fire-model",
        type=str,
        default=None,
        help="Path to trained fire adversary model to train rescuers against (default: None)",
    )
    return parser.parse_args()


def train(args):
    """Initializes and runs PPO training."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70)
    print("  RADON Multi-Agent Rescue Training")
    print("=" * 70)
    print(f"  Preset:         {args.preset.upper()}")
    print(f"  Run Name:       {args.run_name}")
    print(f"  Target Steps:   {args.timesteps:,}")
    print(f"  Compute Device: {device.upper()} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")
    print(f"  Seed:           {args.seed} ({'FIXED single map' if args.fixed_map else 'PROCEDURAL fresh maps every episode'})")
    print(f"  PPO Hparams:    n_steps={args.n_steps}, batch_size={args.batch_size}, lr={args.lr}, ent_coef={args.ent_coef}")
    print("=" * 70)

    # 1. Environment creation based on preset (fresh maps by default for generalization)
    cfg_seed = args.seed if args.fixed_map else None
    if args.preset == "sanity":
        config = EnvConfig.sanity(seed=cfg_seed)
    elif args.preset == "training_day4_base":
        config = EnvConfig.training_day4_base(seed=cfg_seed)
    elif args.preset == "training_day3":
        config = EnvConfig.training_day3(seed=cfg_seed)
    else:
        config = EnvConfig.training_day2(seed=cfg_seed)

    fc_factory = None
    if args.fire_model and Path(args.fire_model).exists():
        from env.fire_controller import PolicyFireController
        print(f"  Adversary Fire: {args.fire_model}")
        fc_factory = lambda: PolicyFireController(model_path=args.fire_model)

    vec_env = make_rescue_vec_env(
        config=config,
        num_vec_envs=args.num_vec_envs,
        num_cpus=1,
        seed=args.seed,
        fire_controller_factory=fc_factory,
    )
    print(f"  Vec Envs:       {args.num_vec_envs} building copies ({vec_env.num_envs} agents)")

    # 2. Policy configuration
    policy_kwargs = dict(
        features_extractor_class=RescueExtractor,
        features_extractor_kwargs=dict(features_dim=160),
        net_arch=dict(pi=[128, 64], vf=[128, 64]),
    )

    if args.resume and Path(args.resume).exists():
        print(f"  Resuming from:  {args.resume}")
        model = PPO.load(
            args.resume,
            env=vec_env,
            learning_rate=args.lr,
            ent_coef=args.ent_coef,
            device=device,
            seed=args.seed,
        )
    else:
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
            ent_coef=args.ent_coef,
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
        eval_episodes=args.eval_episodes,
        config=config,
        log_dir="metrics",
        run_name=args.run_name,
        fire_controller_factory=fc_factory,
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
    save_path = agents_dir / f"ppo_{args.run_name}.zip"
    model.save(save_path)
    print(f"Model saved to: {save_path}")

    # Final post-training evaluation across train and held-out test seeds
    print("\n[Post-Training Final Evaluation]")
    final_metrics = metrics_callback._evaluate_policy(model=model, num_episodes=20)
    print(
        f"  Train Success:         {final_metrics['train_success']*100:5.1f}%\n"
        f"  Held-Out Test Success: {final_metrics['test_success']*100:5.1f}%\n"
        f"  Train Rescued:         {final_metrics['train_rescued']:4.2f}/{config.n_hostages}\n"
        f"  Held-Out Test Rescued: {final_metrics['test_rescued']:4.2f}/{config.n_hostages}\n"
        f"  Generalization Gap:    {final_metrics['gap_success']*100:+5.1f}%\n"
        f"  Mean Surviving Agents: {final_metrics['mean_rescuers_surviving']:3.1f}/{config.n_rescuers}"
    )

    vec_env.close()
    return model, save_path


if __name__ == "__main__":
    args = parse_args()
    train(args)
