"""PettingZoo ParallelEnv implementation for RADON rescue simulation."""

from collections import deque
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple
import gymnasium
from gymnasium import spaces
import numpy as np
from pettingzoo.utils.env import ParallelEnv

from env.config import EnvConfig
from env.mapgen import BuildingMap, generate_building
from env.tiles import (
    ACTION_DELTAS,
    Action,
    Hostage,
    Rescuer,
    Tile,
    NUM_OBS_CHANNELS,
    CHANNEL_WALL,
    CHANNEL_FIRE,
    CHANNEL_HOSTAGE,
    CHANNEL_EXIT,
    CHANNEL_STAIRS,
    CHANNEL_OTHER_RESCUER,
)


class RadonRescueEnv(ParallelEnv):
    """Multi-agent rescue environment implementing the PettingZoo ParallelEnv API."""

    metadata = {
        "render_modes": ["ansi", "human"],
        "name": "radon_rescue_v0",
    }

    def __init__(self, config: Optional[EnvConfig] = None, render_mode: Optional[str] = None):
        """Initializes RADON rescue environment with the given configuration."""
        super().__init__()
        self.config = config if config is not None else EnvConfig()
        self.render_mode = render_mode

        self.possible_agents: List[str] = [
            f"rescuer_{i}" for i in range(self.config.n_rescuers)
        ]
        self.agents: List[str] = []

        # Internal state
        self.building_map: Optional[BuildingMap] = None
        self.grids: np.ndarray = np.empty((0, 0, 0), dtype=np.int32)
        self.rescuers: Dict[str, Rescuer] = {}
        self.hostages: List[Hostage] = []
        self.fire_positions: List[Tuple[int, int, int]] = []
        self._exit_dist_maps: Dict[int, np.ndarray] = {}
        self._hostage_dist_maps: Dict[int, np.ndarray] = {}
        self.current_step: int = 0
        self.np_random: np.random.Generator = np.random.default_rng(self.config.seed)

    @lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Dict:
        """Returns the cached observation space for the specified agent."""
        view_dim = self.config.view_size
        return spaces.Dict(
            {
                "grid": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(NUM_OBS_CHANNELS, view_dim, view_dim),
                    dtype=np.float32,
                ),
                "self": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(6,),
                    dtype=np.float32,
                ),
            }
        )

    @lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.Discrete:
        """Returns the cached discrete action space for the specified agent."""
        return spaces.Discrete(len(Action))

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, Any]]]:
        """Resets the environment to a new procedural building state."""
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        self.current_step = 0
        self.agents = list(self.possible_agents)

        # Procedural map generation
        self.building_map = generate_building(self.config, seed=seed)
        self.grids = np.copy(self.building_map.grids)

        # Initialize hostage entities
        self.hostages = []
        for idx, (f, r, c) in enumerate(self.building_map.hostages):
            self.hostages.append(
                Hostage(id=idx, floor=f, row=r, col=c, rescued=False, carried_by=None)
            )

        # Initialize fire positions list
        self.fire_positions = list(self.building_map.fires)

        # Initialize rescuer entities at designated spawns
        self.rescuers = {}
        for idx, agent_id in enumerate(self.possible_agents):
            sf, sr, sc = self.building_map.rescuer_spawns[idx % len(self.building_map.rescuer_spawns)]
            self.rescuers[agent_id] = Rescuer(
                id=agent_id,
                floor=sf,
                row=sr,
                col=sc,
                carrying=False,
                alive=True,
            )

        # Precompute BFS distance maps for potential shaping
        self._recompute_shaping_maps(update_exits=True)

        observations = {agent: self._get_observation(agent) for agent in self.agents}
        infos = {agent: self._get_info(agent) for agent in self.agents}

        return observations, infos

    def _get_observation(self, agent_id: str) -> Dict[str, np.ndarray]:
        """Constructs egocentric 7x7 local grid channels and self feature vector."""
        rescuer = self.rescuers[agent_id]
        rad = self.config.view_radius
        view_dim = self.config.view_size

        grid_obs = np.zeros((NUM_OBS_CHANNELS, view_dim, view_dim), dtype=np.float32)

        # If rescuer is dead, return all-wall obs
        if not rescuer.alive:
            grid_obs[CHANNEL_WALL, :, :] = 1.0
            self_feat = np.zeros(6, dtype=np.float32)
            return {"grid": grid_obs, "self": self_feat}

        curr_floor = rescuer.floor
        for dr_idx, dr in enumerate(range(-rad, rad + 1)):
            for dc_idx, dc in enumerate(range(-rad, rad + 1)):
                r = rescuer.row + dr
                c = rescuer.col + dc

                if (
                    r < 0
                    or r >= self.config.grid_h
                    or c < 0
                    or c >= self.config.grid_w
                ):
                    # Out of bounds treated as WALL
                    grid_obs[CHANNEL_WALL, dr_idx, dc_idx] = 1.0
                else:
                    tile_val = self.grids[curr_floor, r, c]
                    if tile_val == Tile.WALL:
                        grid_obs[CHANNEL_WALL, dr_idx, dc_idx] = 1.0
                    elif tile_val == Tile.FIRE:
                        grid_obs[CHANNEL_FIRE, dr_idx, dc_idx] = 1.0
                    elif tile_val == Tile.HOSTAGE:
                        grid_obs[CHANNEL_HOSTAGE, dr_idx, dc_idx] = 1.0
                    elif tile_val == Tile.EXIT:
                        grid_obs[CHANNEL_EXIT, dr_idx, dc_idx] = 1.0
                    elif tile_val == Tile.STAIRS:
                        grid_obs[CHANNEL_STAIRS, dr_idx, dc_idx] = 1.0

                    # Check for other living rescuers occupying this cell
                    for other_id, other in self.rescuers.items():
                        if (
                            other_id != agent_id
                            and other.alive
                            and other.floor == curr_floor
                            and other.row == r
                            and other.col == c
                        ):
                            grid_obs[CHANNEL_OTHER_RESCUER, dr_idx, dc_idx] = 1.0

        # Self vector: [carrying, floor_norm, steps_left_norm, row_norm, col_norm, hostages_remaining_norm]
        carrying_val = 1.0 if rescuer.carrying else 0.0
        floor_norm = float(rescuer.floor) / max(1, (self.config.floors - 1))
        steps_left = max(0, self.config.max_steps - self.current_step)
        steps_left_norm = float(steps_left) / float(self.config.max_steps)
        row_norm = float(rescuer.row) / max(1, (self.config.grid_h - 1))
        col_norm = float(rescuer.col) / max(1, (self.config.grid_w - 1))
        unrescued_count = sum(1 for h in self.hostages if not h.rescued)
        hostages_rem_norm = float(unrescued_count) / max(1, self.config.n_hostages)

        self_feat = np.array(
            [
                carrying_val,
                floor_norm,
                steps_left_norm,
                row_norm,
                col_norm,
                hostages_rem_norm,
            ],
            dtype=np.float32,
        )

        return {"grid": grid_obs, "self": self_feat}

    def _get_info(self, agent_id: str) -> Dict[str, Any]:
        """Returns diagnostic info for an agent."""
        rescuer = self.rescuers[agent_id]
        return {
            "alive": rescuer.alive,
            "carrying": rescuer.carrying,
            "floor": rescuer.floor,
            "row": rescuer.row,
            "col": rescuer.col,
            "hostages_rescued": sum(1 for h in self.hostages if h.rescued),
            "step": self.current_step,
        }

    def _drop_carried_hostage_on_death(self, rescuer: Rescuer) -> None:
        """Drops carried hostage onto the floor when a rescuer dies so episodes stay winnable."""
        if not rescuer.carrying:
            return

        rescuer.carrying = False
        carried_hostage = None
        for h in self.hostages:
            if h.carried_by == rescuer.id:
                carried_hostage = h
                break

        if carried_hostage is None:
            return

        f, r, c = rescuer.floor, rescuer.row, rescuer.col
        drop_pos = None

        # Prefer rescuer's tile unless it is FIRE, EXIT, or WALL
        if self.grids[f, r, c] not in (Tile.FIRE, Tile.EXIT, Tile.WALL):
            drop_pos = (f, r, c)
        else:
            # Search nearest valid EMPTY neighbor
            neighbors = []
            for dr, dc in (
                (-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (-1, 1), (1, -1), (1, 1),
            ):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.config.grid_h and 0 <= nc < self.config.grid_w:
                    if self.grids[f, nr, nc] == Tile.EMPTY:
                        neighbors.append((abs(dr) + abs(dc), nr, nc))
            if neighbors:
                neighbors.sort(key=lambda x: x[0])
                drop_pos = (f, neighbors[0][1], neighbors[0][2])

        if drop_pos is not None:
            df, dr, dc = drop_pos
            self.grids[df, dr, dc] = Tile.HOSTAGE
            carried_hostage.floor = df
            carried_hostage.row = dr
            carried_hostage.col = dc
            carried_hostage.carried_by = None
            carried_hostage.rescued = False
        else:
            # No valid empty neighbor found: mark hostage lost
            carried_hostage.carried_by = None
            carried_hostage.rescued = False

    def _spread_fire(self) -> None:
        """Spreads fire to adjacent empty tiles with fire_spread_prob chance."""
        if not self.config.fire_enabled:
            return
        if self.current_step < self.config.fire_spread_start_step:
            return
        if len(self.fire_positions) >= self.config.fire_max_tiles:
            return

        existing_fires = list(self.fire_positions)
        for ff, fr, fc in existing_fires:
            if len(self.fire_positions) >= self.config.fire_max_tiles:
                break
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nfr, nfc = fr + dr, fc + dc
                if 0 <= nfr < self.config.grid_h and 0 <= nfc < self.config.grid_w:
                    target_tile = self.grids[ff, nfr, nfc]
                    # Fire ONLY spreads onto EMPTY tiles (never WALL, FIRE, EXIT, HOSTAGE, STAIRS)
                    if target_tile == Tile.EMPTY:
                        if self.np_random.random() < self.config.fire_spread_prob:
                            self.grids[ff, nfr, nfc] = Tile.FIRE
                            self.fire_positions.append((ff, nfr, nfc))
                            if len(self.fire_positions) >= self.config.fire_max_tiles:
                                break

    def _recompute_shaping_maps(self, update_exits: bool = True) -> None:
        """Precomputes or updates BFS shortest-path distance maps for potential shaping."""
        if not self.config.enable_shaping or not self.building_map:
            return
        if update_exits or not hasattr(self, "_exit_dist_maps") or not self._exit_dist_maps:
            self._exit_dist_maps = {}
            for f in range(self.config.floors):
                exit_targets = [(er, ec) for ef, er, ec in self.building_map.exits if ef == f]
                self._exit_dist_maps[f] = self._compute_bfs_dist_map(f, exit_targets)

        self._hostage_dist_maps = {}
        for f in range(self.config.floors):
            hostage_targets = [
                (h.row, h.col)
                for h in self.hostages
                if not h.rescued and h.carried_by is None and h.floor == f
            ]
            self._hostage_dist_maps[f] = self._compute_bfs_dist_map(f, hostage_targets)

    def _compute_bfs_dist_map(self, floor: int, targets: List[Tuple[int, int]]) -> np.ndarray:
        """Computes shortest path BFS grid distance map from given targets respecting walls."""
        h, w = self.config.grid_h, self.config.grid_w
        dist_map = np.full((h, w), fill_value=999, dtype=np.int32)
        if not targets:
            return dist_map

        queue: deque[Tuple[int, int]] = deque()
        for tr, tc in targets:
            if 0 <= tr < h and 0 <= tc < w:
                dist_map[tr, tc] = 0
                queue.append((tr, tc))

        while queue:
            r, c = queue.popleft()
            d = dist_map[r, c]
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    if self.grids[floor, nr, nc] != Tile.WALL and dist_map[nr, nc] > d + 1:
                        dist_map[nr, nc] = d + 1
                        queue.append((nr, nc))
        return dist_map

    def step(
        self, actions: Dict[str, int]
    ) -> Tuple[
        Dict[str, Dict[str, np.ndarray]],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict[str, Any]],
    ]:
        """Executes one step in the environment for all active agents."""
        if not self.agents:
            return {}, {}, {}, {}, {}

        self.current_step += 1
        active_stepping_agents = list(self.agents)

        rewards: Dict[str, float] = {agent: 0.0 for agent in active_stepping_agents}
        terminations: Dict[str, bool] = {agent: False for agent in active_stepping_agents}
        truncations: Dict[str, bool] = {agent: False for agent in active_stepping_agents}
        infos: Dict[str, Dict[str, Any]] = {}

        new_hostage_delivered = False
        hostage_state_changed = False

        # --- 1. Process Rescuer Actions ---
        for agent in active_stepping_agents:
            rescuer = self.rescuers[agent]
            if not rescuer.alive:
                terminations[agent] = True
                continue

            # Base step time penalty
            rewards[agent] += self.config.step_reward

            action_val = actions.get(agent, Action.STAY)
            try:
                action = Action(action_val)
            except ValueError:
                action = Action.STAY

            if action in (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT):
                dr, dc = ACTION_DELTAS[action]
                nr, nc = rescuer.row + dr, rescuer.col + dc

                # Potential-based BFS distance shaping (before movement)
                prev_dist = None
                if self.config.enable_shaping and self.building_map:
                    f = rescuer.floor
                    if rescuer.carrying:
                        if f in self._exit_dist_maps:
                            prev_dist = int(self._exit_dist_maps[f][rescuer.row, rescuer.col])
                    else:
                        if f in self._hostage_dist_maps:
                            prev_dist = int(self._hostage_dist_maps[f][rescuer.row, rescuer.col])

                # Check boundaries and walls
                if (
                    nr < 0
                    or nr >= self.config.grid_h
                    or nc < 0
                    or nc >= self.config.grid_w
                    or self.grids[rescuer.floor, nr, nc] == Tile.WALL
                ):
                    # Wall collision / boundary -> no-op
                    pass
                else:
                    rescuer.row = nr
                    rescuer.col = nc
                    # If carrying a hostage, update carried hostage position
                    if rescuer.carrying:
                        for h in self.hostages:
                            if h.carried_by == agent:
                                h.row = nr
                                h.col = nc

                    # Reward shaping delta using true shortest BFS path
                    if prev_dist is not None and prev_dist < 999:
                        f = rescuer.floor
                        if rescuer.carrying:
                            curr_dist = int(self._exit_dist_maps[f][rescuer.row, rescuer.col]) if f in self._exit_dist_maps else 999
                        else:
                            curr_dist = int(self._hostage_dist_maps[f][rescuer.row, rescuer.col]) if f in self._hostage_dist_maps else 999
                        if curr_dist < 999:
                            rewards[agent] += (prev_dist - curr_dist) * self.config.shaping_coef

                    # Step onto FIRE kills agent
                    if self.grids[rescuer.floor, nr, nc] == Tile.FIRE:
                        rescuer.alive = False
                        rewards[agent] += self.config.death_reward
                        terminations[agent] = True
                        self._drop_carried_hostage_on_death(rescuer)
                        hostage_state_changed = True
                    else:
                        # Auto Pick up: stepping onto HOSTAGE tile while not carrying
                        if not rescuer.carrying and self.grids[rescuer.floor, nr, nc] == Tile.HOSTAGE:
                            for h in self.hostages:
                                if (
                                    not h.rescued
                                    and h.carried_by is None
                                    and h.floor == rescuer.floor
                                    and h.row == nr
                                    and h.col == nc
                                ):
                                    rescuer.carrying = True
                                    h.carried_by = agent
                                    rewards[agent] += self.config.pickup_reward
                                    self.grids[rescuer.floor, nr, nc] = Tile.EMPTY
                                    hostage_state_changed = True
                                    break

                        # Auto Deliver: stepping onto EXIT tile while carrying
                        elif rescuer.carrying and self.grids[rescuer.floor, nr, nc] == Tile.EXIT:
                            rescuer.carrying = False
                            for h in self.hostages:
                                if h.carried_by == agent:
                                    h.carried_by = None
                                    h.rescued = True
                                    rewards[agent] += self.config.delivery_reward
                                    new_hostage_delivered = True
                                    hostage_state_changed = True
                                    break

            elif action == Action.INTERACT:
                curr_f = rescuer.floor
                curr_r = rescuer.row
                curr_c = rescuer.col
                tile_val = self.grids[curr_f, curr_r, curr_c]

                # A. Use STAIRS -> transition to the other floor (if multi-floor)
                if tile_val == Tile.STAIRS and self.config.floors > 1:
                    target_floor = 1 - curr_f if self.config.floors == 2 else (curr_f + 1) % self.config.floors
                    rescuer.floor = target_floor
                    if rescuer.carrying:
                        for h in self.hostages:
                            if h.carried_by == agent:
                                h.floor = target_floor

                # B. Manual Pick up fallback
                elif not rescuer.carrying and tile_val == Tile.HOSTAGE:
                    for h in self.hostages:
                        if (
                            not h.rescued
                            and h.carried_by is None
                            and h.floor == curr_f
                            and h.row == curr_r
                            and h.col == curr_c
                        ):
                            rescuer.carrying = True
                            h.carried_by = agent
                            rewards[agent] += self.config.pickup_reward
                            self.grids[curr_f, curr_r, curr_c] = Tile.EMPTY
                            hostage_state_changed = True
                            break

                # C. Manual Drop at EXIT fallback
                elif rescuer.carrying and tile_val == Tile.EXIT:
                    rescuer.carrying = False
                    for h in self.hostages:
                        if h.carried_by == agent:
                            h.carried_by = None
                            h.rescued = True
                            rewards[agent] += self.config.delivery_reward
                            new_hostage_delivered = True
                            hostage_state_changed = True
                            break

        if hostage_state_changed:
            self._recompute_shaping_maps(update_exits=False)

        # --- 2. Shared Team Completion Bonus ---
        all_hostages_rescued = all(h.rescued for h in self.hostages)
        if new_hostage_delivered and all_hostages_rescued:
            for agent in active_stepping_agents:
                rewards[agent] += self.config.team_completion_reward

        # --- 3. Fire Dynamics (via _spread_fire) ---
        self._spread_fire()

        # Check if spreading fire engulfed any living rescuers
        for agent in active_stepping_agents:
            rescuer = self.rescuers[agent]
            if rescuer.alive and self.grids[rescuer.floor, rescuer.row, rescuer.col] == Tile.FIRE:
                rescuer.alive = False
                rewards[agent] += self.config.death_reward
                terminations[agent] = True
                self._drop_carried_hostage_on_death(rescuer)

        # --- 4. Episode Termination & Truncation Checks ---
        all_rescuers_dead = not any(r.alive for r in self.rescuers.values())
        is_truncated = self.current_step >= self.config.max_steps

        for agent in active_stepping_agents:
            rescuer = self.rescuers[agent]
            if not rescuer.alive or all_hostages_rescued or all_rescuers_dead:
                terminations[agent] = True
            if is_truncated:
                truncations[agent] = True

            infos[agent] = self._get_info(agent)

        # Observations must be provided for all stepping agents this turn
        observations = {agent: self._get_observation(agent) for agent in active_stepping_agents}

        # Update active agents list: remove all terminated/truncated agents
        self.agents = [
            agent
            for agent in active_stepping_agents
            if not (terminations[agent] or truncations[agent])
        ]

        return observations, rewards, terminations, truncations, infos

    def render(self) -> Optional[str]:
        """Renders ASCII representation of building floors."""
        symbols = {
            Tile.EMPTY: ".",
            Tile.WALL: "#",
            Tile.STAIRS: "S",
            Tile.EXIT: "E",
            Tile.HOSTAGE: "H",
            Tile.FIRE: "F",
        }

        output_lines = [f"=== Step {self.current_step} ==="]
        for f in range(self.config.floors):
            output_lines.append(f"--- Floor {f} ---")
            grid_chars = []
            for r in range(self.config.grid_h):
                row_chars = []
                for c in range(self.config.grid_w):
                    rescuer_on_cell = None
                    for ag_id, rescuer in self.rescuers.items():
                        if rescuer.alive and rescuer.floor == f and rescuer.row == r and rescuer.col == c:
                            idx_str = ag_id.split("_")[-1]
                            rescuer_on_cell = f"R{idx_str}"
                            break

                    if rescuer_on_cell:
                        row_chars.append(rescuer_on_cell)
                    else:
                        tile = Tile(self.grids[f, r, c])
                        row_chars.append(f" {symbols.get(tile, '?')}")
                grid_chars.append("".join(row_chars))
            output_lines.append("\n".join(grid_chars))

        output_str = "\n".join(output_lines)
        if self.render_mode == "human":
            print(output_str)
        return output_str

    def close(self) -> None:
        """Cleans up resources."""
        pass
