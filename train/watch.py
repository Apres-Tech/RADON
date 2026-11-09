"""Visual frame-by-frame inspector to watch and diagnose trained RADON agent behaviors."""

import argparse
from pathlib import Path
import time
import numpy as np
import torch
from stable_baselines3 import PPO

from env.config import EnvConfig
from env.rescue_env import RadonRescueEnv
from env.tiles import Action, Tile


def watch_episode(
    model_path: str | Path | None = None,
    preset: str = "training_day3",
    seed: int = 42,
    deterministic: bool = False,
    delay: float = 0.1,
    max_steps: int | None = None,
):
    """Runs a single episode and displays ASCII rendering and agent diagnostics at each step."""
    if preset == "sanity":
        config = EnvConfig.sanity(seed=seed)
    elif preset == "training_day4_base":
        config = EnvConfig.training_day4_base(seed=seed)
    elif preset == "training_day3":
        config = EnvConfig.training_day3(seed=seed)
    else:
        config = EnvConfig.training_day2(seed=seed)

    if max_steps is not None:
        config.max_steps = max_steps

    env = RadonRescueEnv(config=config)
    obs, infos = env.reset(seed=seed)

    model = None
    if model_path is not None and Path(model_path).exists():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = PPO.load(model_path, device=device)
        print(f"Loaded policy from: {model_path} (on {device})")
    else:
        print("No model path specified or not found; running RANDOM policy.")

    print("=" * 65)
    print(f"  RADON Behavior Inspector (Preset: {preset}, Seed: {seed})")
    print(f"  Grid: {config.grid_h}x{config.grid_w}, Rescuers: {config.n_rescuers}, Hostages: {config.n_hostages}")
    print("=" * 65)

    print("\n--- Initial State (Step 0) ---")
    env.render()

    step_count = 0
    total_rewards = {agent: 0.0 for agent in env.possible_agents}
    delivered_count = 0

    while env.agents:
        step_count += 1
        actions = {}
        for agent_id in env.agents:
            if model is not None:
                agent_obs = {
                    k: np.expand_dims(v, axis=0) for k, v in obs[agent_id].items()
                }
                action, _ = model.predict(agent_obs, deterministic=deterministic)
                actions[agent_id] = int(action[0])
            else:
                actions[agent_id] = int(env.action_space(agent_id).sample())

        action_desc = ", ".join(f"{aid}: {Action(act).name}" for aid, act in sorted(actions.items()))

        # Check carrying state before step
        prev_carrying = {aid: env.rescuers[aid].carrying for aid in env.agents}

        obs, rewards, terminations, truncations, infos = env.step(actions)

        # Diagnose notable events
        events = []
        for aid, rew in rewards.items():
            total_rewards[aid] += rew
            rescuer = env.rescuers[aid]
            if not prev_carrying[aid] and rescuer.carrying:
                events.append(f"PICKUP by {aid} at ({rescuer.row}, {rescuer.col})")
            elif prev_carrying[aid] and not rescuer.carrying:
                events.append(f"DELIVER by {aid} at EXIT ({rescuer.row}, {rescuer.col})! (+10)")
                delivered_count += 1
            if not rescuer.alive:
                events.append(f"DEATH: {aid} stepped into fire!")

        print(f"\n--- Step {step_count} | Actions: [{action_desc}] ---")
        if events:
            print("  EVENTS: " + " | ".join(events))
        env.render()

        # Print agent statuses
        status_strs = []
        for aid in sorted(env.possible_agents):
            r = env.rescuers[aid]
            c_tag = "CARRYING" if r.carrying else "free"
            a_tag = "ALIVE" if r.alive else "DEAD"
            status_strs.append(f"{aid}: F{r.floor}({r.row},{r.col}) [{a_tag}, {c_tag}, rew={total_rewards[aid]:+.1f}]")
        print("  " + " | ".join(status_strs[:2]))
        if len(status_strs) > 2:
            print("  " + " | ".join(status_strs[2:]))

        if delay > 0:
            time.sleep(delay)

    rescued_total = sum(1 for h in env.hostages if h.rescued)
    surviving_total = sum(1 for r in env.rescuers.values() if r.alive)
    all_evacuated = (rescued_total == config.n_hostages)

    print("\n" + "=" * 65)
    print(f"  Episode Terminated at Step {step_count}")
    print(f"  Result:              {'ALL RESCUED (SUCCESS!)' if all_evacuated else 'INCOMPLETE'}")
    print(f"  Hostages Rescued:    {rescued_total} / {config.n_hostages}")
    print(f"  Rescuers Surviving:  {surviving_total} / {config.n_rescuers}")
    print(f"  Fire Deaths:         {env.fire_deaths}")
    print(f"  Total Return:        {sum(total_rewards.values()):+.2f}")
    print("=" * 65)
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watch RADON episode in ASCII")
    parser.add_argument("--model-path", type=str, default=None, help="Model checkpoint path")
    parser.add_argument("--preset", type=str, default="training_day4_base", choices=["training_day4_base", "training_day3", "training_day2", "sanity"])
    parser.add_argument("--seed", type=int, default=42, help="Map seed")
    parser.add_argument("--deterministic", action="store_true", help="Deterministic policy")
    parser.add_argument("--delay", type=float, default=0.05, help="Delay between steps in seconds")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max steps")
    args = parser.parse_args()
    watch_episode(
        model_path=args.model_path,
        preset=args.preset,
        seed=args.seed,
        deterministic=args.deterministic,
        delay=args.delay,
        max_steps=args.max_steps,
    )

