"""Unit tests verifying Stage 1 BFS direction and distance observation features."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from env.config import EnvConfig
from env.rescue_env import RadonRescueEnv
from train.extractor import RescueExtractor


def test_observation_space_shapes():
    # 1. Verify 13-dim self vector when obs_bfs_features=True
    cfg_bfs = EnvConfig.showcase(obs_bfs_features=True)
    env_bfs = RadonRescueEnv(config=cfg_bfs)
    obs_space_bfs = env_bfs.observation_space("rescuer_0")
    assert obs_space_bfs["self"].shape == (13,), f"Expected shape (13,), got {obs_space_bfs['self'].shape}"
    assert obs_space_bfs["grid"].shape == (6, 7, 7), f"Expected shape (6, 7, 7), got {obs_space_bfs['grid'].shape}"

    # 2. Verify 6-dim self vector when obs_bfs_features=False (legacy backwards compatibility)
    cfg_legacy = EnvConfig.showcase(obs_bfs_features=False)
    env_legacy = RadonRescueEnv(config=cfg_legacy)
    obs_space_legacy = env_legacy.observation_space("rescuer_0")
    assert obs_space_legacy["self"].shape == (6,), f"Expected shape (6,), got {obs_space_legacy['self'].shape}"
    print("[PASS] observation_space shapes verified for both BFS (13) and legacy (6).")


def test_reset_and_features():
    cfg = EnvConfig.showcase(obs_bfs_features=True, seed=42)
    env = RadonRescueEnv(config=cfg)
    obs, infos = env.reset(seed=42)

    for aid in env.agents:
        agent_obs = obs[aid]
        assert "grid" in agent_obs and "self" in agent_obs
        self_feat = agent_obs["self"]
        assert self_feat.shape == (13,)
        assert self_feat.dtype == np.float32

        # Check ranges
        # [carrying, floor, steps_left, row, col, hostages_rem, up, down, left, right, dist, diff_floor, has_goal]
        deltas = self_feat[6:10]
        for d in deltas:
            assert -1.0 <= d <= 1.0, f"Delta out of bounds: {d}"

        norm_dist = self_feat[10]
        assert 0.0 <= norm_dist <= 1.0, f"Norm dist out of bounds: {norm_dist}"

        goal_diff_floor = self_feat[11]
        assert goal_diff_floor in (0.0, 1.0), f"Invalid goal_diff_floor: {goal_diff_floor}"

        has_goal = self_feat[12]
        assert has_goal in (0.0, 1.0), f"Invalid has_goal: {has_goal}"

    print("[PASS] reset() and observation features bounds verified.")


def test_size_invariance():
    """Verify identical observation shapes across 10x10, 14x14, and 20x20 maps."""
    sizes = [10, 14, 20]
    for s in sizes:
        cfg = EnvConfig.showcase(grid_size=s, obs_bfs_features=True, seed=123)
        env = RadonRescueEnv(config=cfg)
        obs, _ = env.reset(seed=123)
        for aid in env.agents:
            assert obs[aid]["grid"].shape == (6, 7, 7)
            assert obs[aid]["self"].shape == (13,)
    print("[PASS] Size-invariance verified: exact same (6,7,7) grid and (13,) self vector across 10x10, 14x14, 20x20.")


def test_rescue_extractor():
    """Verify RescueExtractor forward pass on batch of observations."""
    cfg = EnvConfig.showcase(obs_bfs_features=True, seed=42)
    env = RadonRescueEnv(config=cfg)
    obs_space = env.observation_space("rescuer_0")

    extractor = RescueExtractor(obs_space, features_dim=160)
    obs, _ = env.reset(seed=42)

    batch_grid = torch.as_tensor(np.stack([obs[aid]["grid"] for aid in env.agents]), dtype=torch.float32)
    batch_self = torch.as_tensor(np.stack([obs[aid]["self"] for aid in env.agents]), dtype=torch.float32)

    out = extractor({"grid": batch_grid, "self": batch_self})
    assert out.shape == (len(env.agents), 160), f"Expected shape ({len(env.agents)}, 160), got {out.shape}"
    print("[PASS] RescueExtractor forward pass verified with shape (N, 160).")


if __name__ == "__main__":
    test_observation_space_shapes()
    test_reset_and_features()
    test_size_invariance()
    test_rescue_extractor()
    print("\nALL STAGE 1 BFS OBSERVATION UNIT TESTS PASSED!")
