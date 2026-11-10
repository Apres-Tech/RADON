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

        # 2. Interior walls with WIDE doorways (open office floor plan)
        door_width = 3 if (h >= 16 or w >= 16) else 2
        for f in range(floors):
            # Choose vertical dividing lines
            max_v = max(1, min(2, (w - 6) // 4))
            n_v_walls = rng.integers(1, max_v + 1)
            # Space dividers nicely
            min_col = 3
            max_col = w - 4
            if max_col > min_col:
                v_cols = rng.choice(np.arange(min_col, max_col), size=min(n_v_walls, max_col - min_col), replace=False)
                for vc in v_cols:
                    grids[f, 1 : h - 1, vc] = Tile.WALL
                    # Punch 2-3 WIDE doorways (each 2-3 tiles wide)
                    n_doors = min(3, max(2, (h - 2) // 5))
                    door_candidates = [r for r in range(2, h - 2 - door_width, door_width + 1)]
                    if door_candidates:
                        chosen_starts = rng.choice(door_candidates, size=min(n_doors, len(door_candidates)), replace=False)
                        for d_start in chosen_starts:
                            grids[f, d_start : d_start + door_width, vc] = Tile.EMPTY
                    else:
                        grids[f, 1 : h - 1, vc] = Tile.EMPTY

            # Choose horizontal dividing lines
            max_h = max(1, min(2, (h - 6) // 4))
            n_h_walls = rng.integers(1, max_h + 1)
            min_row = 3
            max_row = h - 4
            if max_row > min_row:
                h_rows = rng.choice(np.arange(min_row, max_row), size=min(n_h_walls, max_row - min_row), replace=False)
                for hr in h_rows:
                    grids[f, hr, 1 : w - 1] = Tile.WALL
                    # Punch 2-3 WIDE doorways
                    n_doors = min(3, max(2, (w - 2) // 5))
                    door_candidates = [c for c in range(2, w - 2 - door_width, door_width + 1)]
                    if door_candidates:
                        chosen_starts = rng.choice(door_candidates, size=min(n_doors, len(door_candidates)), replace=False)
                        for d_start in chosen_starts:
                            grids[f, hr, d_start : d_start + door_width] = Tile.EMPTY
                    else:
                        grids[f, hr, 1 : w - 1] = Tile.EMPTY

            # Add a few open architectural pillars (never blocking doorways)
            n_pillars = 2 if (h <= 14 or w <= 14) else 4
            for _ in range(n_pillars):
                pr = rng.integers(3, h - 3)
                pc = rng.integers(3, w - 3)
                # Only place pillar if surrounding area is open
                if np.sum(grids[f, pr - 1 : pr + 2, pc - 1 : pc + 2] == Tile.WALL) <= 1:
                    grids[f, pr, pc] = Tile.WALL

        # 3. Stairs: Same (row, col) across ALL floors (wide/prominent stairwell)
        stairs_list: List[Tuple[int, int]] = []
        if floors > 1:
            potential_stairs = []
            # Prefer central open locations
            mid_r, mid_c = h // 2, w // 2
            for r in range(max(2, mid_r - 4), min(h - 2, mid_r + 4)):
                for c in range(max(2, mid_c - 4), min(w - 2, mid_c + 4)):
                    if all(grids[f, r, c] == Tile.EMPTY for f in range(floors)):
                        potential_stairs.append((r, c))

            if not potential_stairs:
                # Fallback to any empty spot
                for r in range(2, h - 2):
                    for c in range(2, w - 2):
                        if all(grids[f, r, c] == Tile.EMPTY for f in range(floors)):
                            potential_stairs.append((r, c))

            if not potential_stairs:
                continue

            stair_idx = rng.integers(0, len(potential_stairs))
            stair_rc = potential_stairs[stair_idx]
            stairs_list = [stair_rc]
            # If map is large, create a 2-tile wide stair landing
            if w >= 16 and (stair_rc[0], stair_rc[1] + 1) in potential_stairs:
                stairs_list.append((stair_rc[0], stair_rc[1] + 1))

            for s_rc in stairs_list:
                for f in range(floors):
                    grids[f, s_rc[0], s_rc[1]] = Tile.STAIRS

        # 4. Multi-Tile Exit ZONES on Floor 0 (e.g. 2x2 or 2x1 entrance portals)
        exits_list: List[Tuple[int, int, int]] = []
        stairs_set = set(stairs_list)
        # Find candidate 2x2 zones along bottom or top perimeter
        zone_candidates = []
        for r in (h - 3, 1):
            for c in range(2, w - 3, 3):
                zone_coords = [(0, r, c), (0, r + 1, c), (0, r, c + 1), (0, r + 1, c + 1)]
                if all(
                    1 <= zr < h - 1 and 1 <= zc < w - 1
                    and grids[0, zr, zc] != Tile.WALL
                    and (zr, zc) not in stairs_set
                    for (_, zr, zc) in zone_coords
                ):
                    zone_candidates.append(zone_coords)

        if zone_candidates:
            # Pick 1 or 2 exit zones
            n_zones = 2 if (len(zone_candidates) >= 2 and w >= 16) else 1
            chosen_zone_indices = rng.choice(len(zone_candidates), size=n_zones, replace=False)
            for z_idx in chosen_zone_indices:
                for z_coord in zone_candidates[z_idx]:
                    if z_coord not in exits_list:
                        exits_list.append(z_coord)
                        grids[z_coord[0], z_coord[1], z_coord[2]] = Tile.EXIT
        else:
            # Fallback to single-tile exits if tight map
            potential_exits = []
            for r in range(1, h - 1):
                for c in range(1, w - 1):
                    if grids[0, r, c] == Tile.EMPTY and (r, c) not in stairs_set:
                        potential_exits.append((0, r, c))
            if not potential_exits:
                continue
            n_exits = min(4, len(potential_exits))
            exit_indices = rng.choice(len(potential_exits), size=n_exits, replace=False)
            exits_list = [potential_exits[i] for i in exit_indices]
            for ep in exits_list:
                grids[ep[0], ep[1], ep[2]] = Tile.EXIT

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

        # 6. Hostages: Distribute across floors in OPEN, clearly reachable spots (no dead-ends)
        hostage_list = []
        hostages_per_floor = [n_hostages // floors] * floors
        for i in range(n_hostages % floors):
            hostages_per_floor[i] += 1

        failed_hostages = False
        occupied_locs = set(exits_list) | set(rescuer_spawns)
        for f in range(floors):
            floor_empty = []
            open_candidates = []
            for r in range(1, h - 1):
                for c in range(1, w - 1):
                    loc = (f, r, c)
                    if grids[loc] == Tile.EMPTY and loc not in occupied_locs:
                        floor_empty.append(loc)
                        # Check surrounding openness (at least 3 open cardinal neighbors)
                        open_count = sum(
                            1
                            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                            if 0 <= r + dr < h and 0 <= c + dc < w and grids[f, r + dr, c + dc] == Tile.EMPTY
                        )
                        if open_count >= 3:
                            open_candidates.append(loc)

            target_count = hostages_per_floor[f]
            pool = open_candidates if len(open_candidates) >= target_count else floor_empty
            if len(pool) < target_count:
                failed_hostages = True
                break

            chosen = rng.choice(len(pool), size=target_count, replace=False)
            for idx in chosen:
                h_pos = pool[idx]
                hostage_list.append(h_pos)
                occupied_locs.add(h_pos)
                grids[h_pos] = Tile.HOSTAGE

        if failed_hostages or len(hostage_list) != n_hostages:
            continue

        # 7. Initial Fire: Place a few fire tiles away from spawns & exits
        fires_list = []
        if getattr(config, "fire_enabled", True):
            for f in range(floors):
                floor_candidates = []
                for r in range(1, h - 1):
                    for c in range(1, w - 1):
                        loc = (f, r, c)
                        if grids[loc] == Tile.EMPTY and loc not in occupied_locs:
                            # Keep initial fire safely away from spawns on Floor 0
                            min_spawn_dist = min(abs(r - sr) + abs(c - sc) for (_, sr, sc) in rescuer_spawns) if rescuer_spawns else 10
                            if min_spawn_dist >= 4 or f > 0:
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
