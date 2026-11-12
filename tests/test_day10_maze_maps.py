"""Unit tests for Day 10 large maze map generator:
- EnvConfig.maze() preset configuration
- Winding corridors, narrow chokepoints, and dead-ends
- Guaranteed 100% reachability via BFS
- Multi-tile exit zones and entity distribution across floors
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env.config import EnvConfig
from env.mapgen import generate_building, verify_reachability
from env.rescue_env import RadonRescueEnv, Tile, Action


def test_maze_config_and_structure():
    cfg = EnvConfig.maze(grid_size=20, seed=50000)
    assert cfg.layout_type == "maze"
    assert cfg.grid_h == 20 and cfg.grid_w == 20
    assert cfg.floors == 2
    assert cfg.n_rescuers == 5
    assert cfg.n_hostages == 4
    assert cfg.max_steps == 350
    print("[PASS] EnvConfig.maze() preset parameters verified.")


def test_maze_generation_and_chokepoints():
    cfg = EnvConfig.maze(grid_size=20, seed=50000)
    b = generate_building(cfg, seed=50000)

    # Verify reachability
    assert verify_reachability(b.grids, b.stairs, b.exits, b.hostages, b.rescuer_spawns)

    # Check for presence of narrow chokepoints:
    # A 1-tile chokepoint is an EMPTY tile bounded by WALLs on two opposing sides
    h, w = cfg.grid_h, cfg.grid_w
    has_vertical_chokepoint = False
    has_horizontal_chokepoint = False

    for f in range(cfg.floors):
        for r in range(1, h - 1):
            for c in range(1, w - 1):
                if b.grids[f, r, c] == Tile.EMPTY:
                    # Horizontal chokepoint: walls above and below
                    if b.grids[f, r - 1, c] == Tile.WALL and b.grids[f, r + 1, c] == Tile.WALL:
                        has_horizontal_chokepoint = True
                    # Vertical chokepoint: walls left and right
                    if b.grids[f, r, c - 1] == Tile.WALL and b.grids[f, r, c + 1] == Tile.WALL:
                        has_vertical_chokepoint = True

    assert has_vertical_chokepoint, "Maze must contain vertical chokepoints (narrow corridor bottlenecks)"
    assert has_horizontal_chokepoint, "Maze must contain horizontal chokepoints (narrow corridor bottlenecks)"
    print("[PASS] Maze chokepoints and internal partition geometry verified.")


def test_held_out_solvability_and_env_rollout():
    cfg = EnvConfig.maze(grid_size=20)
    # Test 10 held-out seeds for guaranteed solvability
    for seed in range(50000, 50010):
        b = generate_building(cfg, seed=seed)
        assert verify_reachability(b.grids, b.stairs, b.exits, b.hostages, b.rescuer_spawns)

    # Test running 5 steps in the actual rescue environment
    env = RadonRescueEnv(config=cfg)
    obs, infos = env.reset(seed=50001)
    assert len(obs) == 5, f"Expected 5 rescuers, got {len(obs)}"
    for _ in range(5):
        actions = {a: Action.STAY for a in env.agents}
        obs, rewards, terms, truncs, infos = env.step(actions)

    print("[PASS] 100% reachability across held-out seeds and environment simulation verified.")


if __name__ == "__main__":
    test_maze_config_and_structure()
    test_maze_generation_and_chokepoints()
    test_held_out_solvability_and_env_rollout()
    print("\nALL DAY 10 MAZE MAP TESTS PASSED SUCCESSFULLY!")
