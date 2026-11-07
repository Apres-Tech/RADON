"""Procedural multi-floor building generator with reachability verification for RADON."""

from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple
import numpy as np

from env.config import EnvConfig
from env.tiles import Tile


@dataclass
class BuildingMap:
    """Generated multi-floor building map and entity placement coordinates."""

    floors: int
    grid_h: int
    grid_w: int
    grids: np.ndarray  # Shape: (floors, grid_h, grid_w), dtype int32
    stairs: List[Tuple[int, int]]  # (row, col) identical across all floors
    exits: List[Tuple[int, int, int]]  # (floor, row, col), floor == 0
    hostages: List[Tuple[int, int, int]]  # (floor, row, col)
    fires: List[Tuple[int, int, int]]  # (floor, row, col)
    rescuer_spawns: List[Tuple[int, int, int]]  # (floor, row, col)


def verify_reachability(
    grids: np.ndarray,
    stairs: List[Tuple[int, int]],
    exits: List[Tuple[int, int, int]],
    hostages: List[Tuple[int, int, int]],
    rescuer_spawns: List[Tuple[int, int, int]],
) -> bool:
    """Verifies with BFS that all rescuers, stairs, and hostages can reach an exit.

    Navigation moves:
      - 4-directional moves on the same floor (non-wall, non-fire tiles).
      - Inter-floor transition when standing on STAIRS tiles.
    """
    floors, h, w = grids.shape
    if not exits:
        return False

    # Multi-source BFS starting from all exits on floor 0
    queue: deque[Tuple[int, int, int]] = deque()
    visited: Set[Tuple[int, int, int]] = set()

    for exit_loc in exits:
        f, r, c = exit_loc
        if grids[f, r, c] != Tile.WALL and grids[f, r, c] != Tile.FIRE:
            queue.append((f, r, c))
            visited.add((f, r, c))

    stairs_set = set(stairs)

    while queue:
        f, r, c = queue.popleft()

        # Check stair transition across adjacent floors (if stairs exist)
        if (r, c) in stairs_set and grids[f, r, c] == Tile.STAIRS:
            for nf in (f - 1, f + 1):
                if 0 <= nf < floors:
                    nxt = (nf, r, c)
                    if (
                        nxt not in visited
                        and grids[nf, r, c] != Tile.WALL
                        and grids[nf, r, c] != Tile.FIRE
                    ):
                        visited.add(nxt)
                        queue.append(nxt)

        # 4-connected lateral moves
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                nxt = (f, nr, nc)
                if (
                    nxt not in visited
                    and grids[f, nr, nc] != Tile.WALL
                    and grids[f, nr, nc] != Tile.FIRE
                ):
                    visited.add(nxt)
                    queue.append(nxt)

    # Check all hostages are reachable from exits
    for h_loc in hostages:
        if h_loc not in visited:
            return False

    # Check all rescuer spawns are reachable from exits
    for r_loc in rescuer_spawns:
        if r_loc not in visited:
            return False

    # Check all stairs on all floors are reachable from exits (if stairs exist)
    for s_rc in stairs:
        for f in range(floors):
            if (f, s_rc[0], s_rc[1]) not in visited:
                return False

    return True


def generate_building(config: EnvConfig, seed: Optional[int] = None) -> BuildingMap:
    """Procedurally generates a multi-floor building with reachability guarantee.

    Args:
        config: EnvConfig instance containing map dimensions, counts, and tunables.
        seed: Random seed for reproducibility.

    Returns:
        BuildingMap containing tile grids and coordinate lists.
    """
    effective_seed = seed if seed is not None else config.seed
    rng = np.random.default_rng(effective_seed)

    floors = config.floors
    h = config.grid_h
    w = config.grid_w
    n_hostages = config.n_hostages
    n_rescuers = config.n_rescuers

    max_attempts = 200
    for attempt in range(max_attempts):
        grids = np.full((floors, h, w), fill_value=int(Tile.EMPTY), dtype=np.int32)

        # 1. Outer perimeter walls on all floors
        for f in range(floors):
            grids[f, 0, :] = Tile.WALL
            grids[f, h - 1, :] = Tile.WALL
            grids[f, :, 0] = Tile.WALL
            grids[f, :, w - 1] = Tile.WALL

        # 2. Interior walls with doorways
        for f in range(floors):
            # Choose vertical dividing lines
            max_v = max(1, min(3, w - 6))
            n_v_walls = rng.integers(1, max_v + 1)
            v_cols = rng.choice(np.arange(3, w - 3), size=n_v_walls, replace=False)
            for vc in v_cols:
                grids[f, 1 : h - 1, vc] = Tile.WALL
                # Punch 2-3 doorways in each vertical wall
                doors = rng.choice(np.arange(1, h - 1), size=min(h - 2, rng.integers(2, 4)), replace=False)
                grids[f, doors, vc] = Tile.EMPTY

            # Choose horizontal dividing lines
            max_h = max(1, min(3, h - 6))
            n_h_walls = rng.integers(1, max_h + 1)
            h_rows = rng.choice(np.arange(3, h - 3), size=n_h_walls, replace=False)
            for hr in h_rows:
                grids[f, hr, 1 : w - 1] = Tile.WALL
                # Punch 2-3 doorways in each horizontal wall
                doors = rng.choice(np.arange(1, w - 1), size=min(w - 2, rng.integers(2, 4)), replace=False)
                grids[f, hr, doors] = Tile.EMPTY

            # Add a few random pillar blocks
            n_pillars = rng.integers(1, 4) if (h <= 12 or w <= 12) else rng.integers(2, 6)
            for _ in range(n_pillars):
                pr = rng.integers(2, h - 2)
                pc = rng.integers(2, w - 2)
                grids[f, pr, pc] = Tile.WALL

        # 3. Stairs: Same (row, col) across ALL floors (only if floors > 1)
        stairs_list: List[Tuple[int, int]] = []
        if floors > 1:
            potential_stairs = []
            for r in range(2, h - 2):
                for c in range(2, w - 2):
                    if all(grids[f, r, c] == Tile.EMPTY for f in range(floors)):
                        potential_stairs.append((r, c))

            if not potential_stairs:
                continue

            stair_idx = rng.integers(0, len(potential_stairs))
            stair_rc = potential_stairs[stair_idx]
            stairs_list = [stair_rc]
            for f in range(floors):
                grids[f, stair_rc[0], stair_rc[1]] = Tile.STAIRS

        # 4. Exits: ONLY on floor 0 (at least 1 exit, e.g. 1 or 2)
        potential_exits = []
        for r in range(1, h - 1):
            for c in range(1, w - 1):
                if grids[0, r, c] == Tile.EMPTY and (r, c) not in stairs_list:
                    # Prefer positions near border
                    if r == 1 or r == h - 2 or c == 1 or c == w - 2:
                        potential_exits.append((0, r, c))

        if not potential_exits:
            # Fallback to any empty tile on floor 0
            for r in range(1, h - 1):
                for c in range(1, w - 1):
                    if grids[0, r, c] == Tile.EMPTY and (r, c) not in stairs_list:
                        potential_exits.append((0, r, c))

        if not potential_exits:
            continue

        n_exits = min(2, len(potential_exits))
        exit_indices = rng.choice(len(potential_exits), size=n_exits, replace=False)
        exits_list = [potential_exits[i] for i in exit_indices]
        for exit_pos in exits_list:
            grids[exit_pos[0], exit_pos[1], exit_pos[2]] = Tile.EXIT

        # 5. Rescuer Spawns: on floor 0 near exits or on empty reachable tiles
        potential_spawns = []
        for exit_pos in exits_list:
            ef, er, ec = exit_pos
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)):
                nr, nc = er + dr, ec + dc
                if 1 <= nr < h - 1 and 1 <= nc < w - 1:
                    if grids[0, nr, nc] in (Tile.EMPTY, Tile.EXIT):
                        if (0, nr, nc) not in potential_spawns:
                            potential_spawns.append((0, nr, nc))

        # Add remaining empty floor 0 tiles if needed
        for r in range(1, h - 1):
            for c in range(1, w - 1):
                if grids[0, r, c] == Tile.EMPTY and (0, r, c) not in potential_spawns:
                    potential_spawns.append((0, r, c))

        if len(potential_spawns) < n_rescuers:
            continue

        rescuer_spawns = [
            potential_spawns[i]
            for i in rng.choice(len(potential_spawns), size=n_rescuers, replace=False)
        ]

        # 6. Hostages: Distribute across floors on EMPTY tiles
        hostage_list = []
        hostages_per_floor = [n_hostages // floors] * floors
        for i in range(n_hostages % floors):
            hostages_per_floor[i] += 1

        failed_hostages = False
        occupied_locs = set(exits_list) | set(rescuer_spawns)
        for f in range(floors):
            floor_empty = []
            for r in range(1, h - 1):
                for c in range(1, w - 1):
                    loc = (f, r, c)
                    if grids[loc] == Tile.EMPTY and loc not in occupied_locs:
                        floor_empty.append(loc)

            target_count = hostages_per_floor[f]
            if len(floor_empty) < target_count:
                failed_hostages = True
                break

            chosen = rng.choice(len(floor_empty), size=target_count, replace=False)
            for idx in chosen:
                h_pos = floor_empty[idx]
                hostage_list.append(h_pos)
                occupied_locs.add(h_pos)
                grids[h_pos] = Tile.HOSTAGE

        if failed_hostages or len(hostage_list) != n_hostages:
            continue

        # 7. Initial Fire: Place a few fire tiles (not on exits, stairs, hostages, rescuer spawns)
        fires_list = []
        if getattr(config, "fire_enabled", True):
            for f in range(floors):
                floor_candidates = []
                for r in range(1, h - 1):
                    for c in range(1, w - 1):
                        loc = (f, r, c)
                        if grids[loc] == Tile.EMPTY and loc not in occupied_locs:
                            floor_candidates.append(loc)

                if floor_candidates:
                    n_fires = min(len(floor_candidates), rng.integers(1, 3))
                    chosen_fires = rng.choice(len(floor_candidates), size=n_fires, replace=False)
                    for c_idx in chosen_fires:
                        f_pos = floor_candidates[c_idx]
                        fires_list.append(f_pos)
                        occupied_locs.add(f_pos)
                        grids[f_pos] = Tile.FIRE

        # 8. Check Reachability via BFS
        if verify_reachability(grids, stairs_list, exits_list, hostage_list, rescuer_spawns):
            return BuildingMap(
                floors=floors,
                grid_h=h,
                grid_w=w,
                grids=grids,
                stairs=stairs_list,
                exits=exits_list,
                hostages=hostage_list,
                fires=fires_list,
                rescuer_spawns=rescuer_spawns,
            )

    raise RuntimeError(
        f"Failed to generate valid connected building map after {max_attempts} attempts. "
        "Check config constraints."
    )
