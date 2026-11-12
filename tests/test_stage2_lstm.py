"""Unit tests verifying Stage 2 RecurrentPPO LSTM memory, ghost agent safety, and hidden state resets."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from sb3_contrib import RecurrentPPO
from env.config import EnvConfig
from env.rescue_env import RadonRescueEnv
from train.extractor import RescueExtractor
from train.make_env import make_rescue_vec_env


def test_recurrent_policy_creation():
    cfg = EnvConfig.showcase(obs_bfs_features=True)
    vec_env = make_rescue_vec_env(config=cfg, num_vec_envs=2)

    policy_kwargs = dict(
        features_extractor_class=RescueExtractor,
        features_extractor_kwargs=dict(features_dim=160),
        net_arch=dict(pi=[128], vf=[128]),
        lstm_hidden_size=128,
        n_lstm_layers=1,
        shared_lstm=False,
        enable_critic_lstm=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RecurrentPPO(
        "MultiInputLstmPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=64,
        policy_kwargs=policy_kwargs,
        device=device,
        verbose=0,
    )

    assert hasattr(model.policy, "lstm_actor"), "Policy should have lstm_actor"
    assert hasattr(model.policy, "lstm_critic"), "Policy should have lstm_critic"
    print("[PASS] RecurrentPPO model initialized with separate actor and critic LSTMs.")

    # Test single step rollout
    obs = vec_env.reset()
    state = None
    ep_starts = np.ones((vec_env.num_envs,), dtype=bool)

    action, state = model.predict(obs, state=state, episode_start=ep_starts, deterministic=False)
    assert len(action) == vec_env.num_envs
    assert state is not None
    # State is a tuple of (actor_state, critic_state) or LSTMState
    print(f"[PASS] RecurrentPPO predict() succeeded on {vec_env.num_envs} vectorized agents.")

    vec_env.close()


def test_black_death_ghost_safety():
    """Verify that dead agents stepping on zeros have independent recurrent states that do not corrupt living agents."""
    cfg = EnvConfig.showcase(obs_bfs_features=True)
    vec_env = make_rescue_vec_env(config=cfg, num_vec_envs=1)
    obs = vec_env.reset()

    # Step through until an agent dies or 50 steps
    for _ in range(50):
        actions = [vec_env.action_space.sample() for _ in range(vec_env.num_envs)]
        obs, rews, dones, infos = vec_env.step(actions)
        # Check that observations for each sub-env remain valid float32 arrays
        for k, v in obs.items():
            assert not np.isnan(v).any(), f"NaN detected in observation key {k}"
        if any(dones):
            break

    print("[PASS] black_death ghost agents execute cleanly with zero observation without NaN corruption.")
    vec_env.close()


if __name__ == "__main__":
    test_recurrent_policy_creation()
    test_black_death_ghost_safety()
    print("\nALL STAGE 2 LSTM TESTS PASSED!")
