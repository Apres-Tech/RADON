"""Unit tests for Day 10 combat balance mechanics:
- Rescuer HP & invulnerability frames (i-frames)
- Healing upon hostage delivery
- Fire burnout to Tile.ASH and passability in BFS
- Fire ignition rate limits and no-point-blank safety constraints
- Observation space shapes (7 channels, 14-dim self vector) and RescueExtractor
"""

import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env.config import EnvConfig
from env.fire_controller import CurriculumFireController, get_valid_fire_frontier
from env.rescue_env import RadonRescueEnv, Tile, Action
from env.tiles import CHANNEL_ASH, NUM_OBS_CHANNELS
from train.extractor import RescueExtractor


def test_observation_space_and_extractor():
    cfg = EnvConfig.showcase()
    env = RadonRescueEnv(config=cfg)
    obs_space = env.observation_space("rescuer_0")

    assert obs_space["grid"].shape == (7, 7, 7), f"Expected (7, 7, 7), got {obs_space['grid'].shape}"
    assert obs_space["self"].shape == (14,), f"Expected (14,), got {obs_space['self'].shape}"

    # Verify extractor forward pass
    extractor = RescueExtractor(obs_space)
    obs, _ = env.reset(seed=42)
    sample_obs = {
        "grid": torch.tensor(np.expand_dims(obs["rescuer_0"]["grid"], axis=0), dtype=torch.float32),
        "self": torch.tensor(np.expand_dims(obs["rescuer_0"]["self"], axis=0), dtype=torch.float32),
    }
    features = extractor(sample_obs)
    assert features.shape == (1, 160), f"Expected (1, 160), got {features.shape}"
    print("[PASS] observation_space and RescueExtractor verified (7 channels, 14-dim self).")


def test_rescuer_hp_and_iframes():
    cfg = EnvConfig.showcase(seed=42)
    cfg.rescuer_hp = 2
    cfg.invulnerability_steps = 2
    cfg.fire_enabled = False  # Isolate direct step onto fire
    env = RadonRescueEnv(config=cfg)
    obs, _ = env.reset(seed=42)

    agent_id = "rescuer_0"
    r = env.rescuers[agent_id]
    assert r.hp == 2
    assert r.invulnerable_steps == 0

    # Place a fire tile adjacent to rescuer_0
    target_r, target_c = r.row + 1, r.col
    env.grids[r.floor, target_r, target_c] = Tile.FIRE

    # Move DOWN onto the fire
    actions = {a: Action.STAY for a in env.agents}
    actions[agent_id] = Action.DOWN
    obs, rewards, terms, truncs, infos = env.step(actions)

    # Rescuer should have survived with 1 HP remaining!
    assert r.alive, "Rescuer should survive first hit with 1 HP remaining"
    assert r.hp == 1, f"Expected 1 HP, got {r.hp}"
    assert r.invulnerable_steps == 2, f"Expected 2 i-frame steps, got {r.invulnerable_steps}"
    assert env.hp_lost == 1

    # While invulnerable, rescuer takes no damage and dashes back to safe ground
    actions[agent_id] = Action.UP
    obs, rewards, terms, truncs, infos = env.step(actions)
    assert r.alive, "Rescuer should remain alive after dashing out of fire"
    assert r.hp == 1, f"Expected 1 HP, got {r.hp}"
    assert r.invulnerable_steps == 1, f"Expected 1 i-frame step left, got {r.invulnerable_steps}"

    # Next step: wait on safe ground, i-frame decrements to 0
    actions[agent_id] = Action.STAY
    obs, rewards, terms, truncs, infos = env.step(actions)
    assert r.alive
    assert r.hp == 1
    assert r.invulnerable_steps == 0

    # Now with 0 i-frames, stepping onto fire again deals the 2nd fatal damage -> death
    actions[agent_id] = Action.DOWN
    obs, rewards, terms, truncs, infos = env.step(actions)
    assert not r.alive, "Rescuer should be dead after HP drops to 0"
    assert terms[agent_id], "Agent should be terminated upon death"
    print("[PASS] Rescuer HP (surviving 1 hit, i-frames, death at 0 HP) verified.")


def test_healing_on_hostage_delivery():
    cfg = EnvConfig.showcase(seed=42)
    cfg.rescuer_hp = 2
    env = RadonRescueEnv(config=cfg)
    env.reset(seed=42)

    agent_id = "rescuer_0"
    r = env.rescuers[agent_id]
    r.hp = 1  # Wounded rescuer

    # Set up carrying hostage next to an exit
    r.carrying = True
    exit_loc = env.building_map.exits[0]
    r.floor, r.row, r.col = exit_loc[0], exit_loc[1] + 1, exit_loc[2]
    # Associate hostage
    h = env.hostages[0]
    h.carried_by = agent_id
    h.rescued = False

    # Step NORTH onto the EXIT
    actions = {a: Action.STAY for a in env.agents}
    actions[agent_id] = Action.UP
    env.step(actions)

    assert h.rescued, "Hostage should be rescued"
    assert r.hp == 2, f"Expected healed HP = 2, got {r.hp}"
    print("[PASS] Rescuer healing (+1 HP on hostage delivery) verified.")


def test_fire_burnout_to_ash_and_bfs():
    cfg = EnvConfig.showcase(seed=42)
    cfg.fire_lifetime = 3
    cfg.ash_cooldown = 10
    cfg.fire_enabled = True
    env = RadonRescueEnv(config=cfg)
    env.reset(seed=42)

    # Place a single fire tile
    f, r, c = 0, 5, 5
    env.grids[f, r, c] = Tile.FIRE
    env.fire_positions = [(f, r, c)]
    env.fire_ages = {(f, r, c): 0}

    # Step environment 3 times to trigger burnout
    actions = {a: Action.STAY for a in env.agents}
    for step_num in range(3):
        env.step(actions)

    assert env.grids[f, r, c] == Tile.ASH, f"Expected Tile.ASH at (0, 5, 5), got {env.grids[f, r, c]}"
    assert (f, r, c) not in env.fire_positions, "Burned-out tile should be removed from fire_positions"
    assert env.ash_cooldowns.get((f, r, c), 0) > 0, "Ash cooldown should be active"

    # Verify that BFS treats Tile.ASH as passable!
    dist_map = env._compute_multi_floor_bfs_dist_map([(f, r, c)])
    assert dist_map[f, r, c] == 0
    # Adjacent empty tile should have distance 1
    assert dist_map[f, r + 1, c] == 1 or dist_map[f, r - 1, c] == 1
    print("[PASS] Fire burnout to Tile.ASH and BFS passability verified.")


def test_no_point_blank_and_rate_limit():
    cfg = EnvConfig.showcase(seed=42)
    cfg.fire_ignitions_per_step = 1
    cfg.no_point_blank_ignition = True
    env = RadonRescueEnv(config=cfg)
    env.reset(seed=42)

    r = env.rescuers["rescuer_0"]
    # Put fire 2 tiles away from rescuer
    fire_pos = (r.floor, r.row + 2, r.col)
    env.grids[fire_pos[0], fire_pos[1], fire_pos[2]] = Tile.FIRE
    env.fire_positions = [fire_pos]

    # The tile between fire and rescuer is (r.floor, r.row + 1, r.col), which is adjacent to rescuer!
    point_blank_pos = (r.floor, r.row + 1, r.col)
    env.grids[point_blank_pos[0], point_blank_pos[1], point_blank_pos[2]] = Tile.EMPTY

    frontier = get_valid_fire_frontier(
        env.grids,
        env.fire_positions,
        cfg.grid_h,
        cfg.grid_w,
        rescuers=env.rescuers,
        no_point_blank=True,
    )

    assert point_blank_pos not in frontier, "Frontier must NOT contain tile adjacent to rescuer (no point-blank)!"

    # Also test rate limit budget
    ctrl = CurriculumFireController(max_fire_tiles=100)
    ctrl.reset(cfg, env.grids, env.fire_positions, env.np_random)
    ctrl.start_step = 0
    ctrl.base_spread_prob = 1.0  # Guarantee spread
    ignitions = ctrl.step(
        current_step=1,
        grids=env.grids,
        fire_positions=env.fire_positions,
        rescuers=env.rescuers,
        hostages=env.hostages,
        np_random=env.np_random,
    )
    assert len(ignitions) <= cfg.fire_ignitions_per_step, (
        f"Ignitions count {len(ignitions)} exceeded budget {cfg.fire_ignitions_per_step}"
    )
    print("[PASS] No point-blank constraint and ignition rate limit budget verified.")


if __name__ == "__main__":
    test_observation_space_and_extractor()
    test_rescuer_hp_and_iframes()
    test_healing_on_hostage_delivery()
    test_fire_burnout_to_ash_and_bfs()
    test_no_point_blank_and_rate_limit()
    print("\nALL DAY 10 BALANCE MECHANICS UNIT TESTS PASSED SUCCESSFULLY!")
