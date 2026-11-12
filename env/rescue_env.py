"""PettingZoo ParallelEnv implementation for RADON rescue simulation."""

from collections import deque
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple
import gymnasium
from gymnasium import spaces
import numpy as np
from pettingzoo.utils.env import ParallelEnv

from env.config import EnvConfig
from env.fire_controller import FireController, CurriculumFireController
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
    CHANNEL_ASH,
)


class RadonRescueEnv(ParallelEnv):
    """Multi-agent rescue environment implementing the PettingZoo ParallelEnv API."""

    metadata = {
        "render_modes": ["ansi", "human"],
        "name": "radon_rescue_v0",
    }

    def __init__(
        self,
        config: Optional[EnvConfig] = None,
        render_mode: Optional[str] = None,
        fire_controller: Optional[FireController] = None,
    ):
        """Initializes RADON rescue environment with the given configuration."""
        super().__init__()
        self.config = config if config is not None else EnvConfig()
        self.render_mode = render_mode
        self.fire_controller = fire_controller if fire_controller is not None else CurriculumFireController()

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
        self._exit_dist_map: Optional[np.ndarray] = None
        self._hostage_dist_map: Optional[np.ndarray] = None
        self.current_step: int = 0
        self.np_random: np.random.Generator = np.random.default_rng(self.config.seed)

    @lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Dict:
        """Returns the cached observation space for the specified agent."""
        view_dim = self.config.view_size
        use_bfs = getattr(self.config, "obs_bfs_features", True)
        self_dim = 14 if use_bfs else 7
        space_dict = {
            "grid": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(NUM_OBS_CHANNELS, view_dim, view_dim),
                dtype=np.float32,
            ),
            "self": spaces.Box(
                low=-1.0 if use_bfs else 0.0,
                high=1.0,
                shape=(self_dim,),
                dtype=np.float32,
            ),
        }
        if getattr(self.config, "obs_centralized_critic", False):
            space_dict["global"] = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(6 * self.config.floors, 10, 10),
                dtype=np.float32,
            )
            space_dict["global_scalars"] = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(4,),
                dtype=np.float32,
            )
        return spaces.Dict(space_dict)

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
        map_seed = seed
        if map_seed is None:
            if self.config.seed is not None:
                map_seed = self.config.seed
            else:
                # Randomize within training seed range [1, 10000) avoiding held-out test seeds [50000, 50100)
                map_seed = int(self.np_random.integers(1, 10000))

        if not hasattr(self, "_distinct_map_seeds"):
            self._distinct_map_seeds = set()
        self._distinct_map_seeds.add(map_seed)

        self.building_map = generate_building(self.config, seed=map_seed)
        self.grids = np.copy(self.building_map.grids)

        # Initialize hostage entities
        self.hostages = []
        for idx, (f, r, c) in enumerate(self.building_map.hostages):
            self.hostages.append(
                Hostage(id=idx, floor=f, row=r, col=c, rescued=False, carried_by=None, evacuated_to_ground=(f == 0))
            )

        # Initialize fire positions list, fire ages, ash cooldowns, and death/damage tracking
        self.fire_positions = list(self.building_map.fires)
        self.fire_ages: Dict[Tuple[int, int, int], int] = {pos: 0 for pos in self.fire_positions}
        self.ash_cooldowns: Dict[Tuple[int, int, int], int] = {}
        self.fire_deaths: int = 0
        self.hp_lost: int = 0
        self.fire_dashes_survived: int = 0
        if self.fire_controller is not None:
            self.fire_controller.reset(self.config, self.grids, self.fire_positions, self.np_random)

        # Initialize rescuer entities at designated spawns
        self.rescuers = {}
        hp_val = getattr(self.config, "rescuer_hp", 2)
        for idx, agent_id in enumerate(self.possible_agents):
            sf, sr, sc = self.building_map.rescuer_spawns[idx % len(self.building_map.rescuer_spawns)]
            self.rescuers[agent_id] = Rescuer(
                id=agent_id,
                floor=sf,
                row=sr,
                col=sc,
                carrying=False,
                alive=True,
                hp=hp_val,
                max_hp=hp_val,
                invulnerable_steps=0,
            )

        # Overtime tracking
        self.in_overtime: bool = False
        self.overtime_steps: int = 0
        self.effective_max_steps: int = self.config.max_steps

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

        # If rescuer is dead, return all-wall obs and zero self-vector
        if not rescuer.alive:
            grid_obs[CHANNEL_WALL, :, :] = 1.0
            self_dim = 14 if getattr(self.config, "obs_bfs_features", True) else 7
            self_feat = np.zeros(self_dim, dtype=np.float32)
            dead_obs = {"grid": grid_obs, "self": self_feat}
            if getattr(self.config, "obs_centralized_critic", False):
                dead_obs["global"] = np.zeros((6 * self.config.floors, 10, 10), dtype=np.float32)
                dead_obs["global_scalars"] = np.zeros(4, dtype=np.float32)
            return dead_obs

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
                    elif tile_val == Tile.ASH:
                        grid_obs[CHANNEL_ASH, dr_idx, dc_idx] = 1.0

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

        # Self vector: [carrying, floor_norm, steps_left_norm, row_norm, col_norm, hostages_remaining_norm, hp_norm]
        carrying_val = 1.0 if rescuer.carrying else 0.0
        floor_norm = float(rescuer.floor) / max(1, (self.config.floors - 1))
        # Rolling effective horizon during carry overtime keeps steps_left positive and in-distribution
        curr_horizon = max(self.config.max_steps, getattr(self, "effective_max_steps", self.config.max_steps))
        steps_left = max(0, curr_horizon - self.current_step)
        steps_left_norm = min(1.0, float(steps_left) / float(self.config.max_steps))
        row_norm = float(rescuer.row) / max(1, (self.config.grid_h - 1))
        col_norm = float(rescuer.col) / max(1, (self.config.grid_w - 1))
        unrescued_count = sum(1 for h in self.hostages if not h.rescued)
        hostages_rem_norm = float(unrescued_count) / max(1, self.config.n_hostages)
        hp_norm = float(rescuer.hp) / float(max(1, getattr(self.config, "rescuer_hp", 2)))

        self_feat = np.array(
            [
                carrying_val,
                floor_norm,
                steps_left_norm,
                row_norm,
                col_norm,
                hostages_rem_norm,
                hp_norm,
            ],
            dtype=np.float32,
        )

        if getattr(self.config, "obs_bfs_features", True):
            if rescuer.carrying:
                active_dist_map = getattr(self, "_exit_dist_map", None)
                active_origin_floor = getattr(self, "_exit_origin_floor", None)
            else:
                active_dist_map = getattr(self, "_hostage_dist_map", None)
                active_origin_floor = getattr(self, "_hostage_origin_floor", None)

            if (
                active_dist_map is not None
                and 0 <= rescuer.floor < active_dist_map.shape[0]
                and 0 <= rescuer.row < active_dist_map.shape[1]
                and 0 <= rescuer.col < active_dist_map.shape[2]
            ):
                cur_d = int(active_dist_map[rescuer.floor, rescuer.row, rescuer.col])
            else:
                cur_d = 999

            if cur_d >= 900:
                # No reachable goal
                delta_up = 0.0
                delta_down = 0.0
                delta_left = 0.0
                delta_right = 0.0
                norm_dist = 1.0
                goal_diff_floor = 0.0
                has_goal = 0.0
            else:
                max_possible_dist = float(self.config.floors * (self.config.grid_h + self.config.grid_w))
                norm_dist = min(1.0, float(cur_d) / max(1.0, max_possible_dist))
                has_goal = 1.0

                target_floor = (
                    active_origin_floor[rescuer.floor, rescuer.row, rescuer.col]
                    if active_origin_floor is not None
                    else rescuer.floor
                )
                goal_diff_floor = 1.0 if target_floor != rescuer.floor else 0.0

                # 4 directional moves: Up (-1, 0), Down (+1, 0), Left (0, -1), Right (0, 1)
                dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                deltas = []
                for dr, dc in dirs:
                    nr, nc = rescuer.row + dr, rescuer.col + dc
                    if nr < 0 or nr >= self.config.grid_h or nc < 0 or nc >= self.config.grid_w:
                        deltas.append(1.0)
                    elif self.grids[rescuer.floor, nr, nc] in (Tile.WALL, Tile.FIRE):
                        deltas.append(1.0)
                    else:
                        nd = int(active_dist_map[rescuer.floor, nr, nc])
                        if nd >= 900:
                            deltas.append(1.0)
                        else:
                            raw_diff = float(nd - cur_d)
                            deltas.append(float(np.clip(raw_diff, -1.0, 1.0)))

                delta_up, delta_down, delta_left, delta_right = deltas

            bfs_feat = np.array(
                [
                    delta_up,
                    delta_down,
                    delta_left,
                    delta_right,
                    norm_dist,
                    goal_diff_floor,
                    has_goal,
                ],
                dtype=np.float32,
            )
            self_feat = np.concatenate([self_feat, bfs_feat])

        obs_dict = {"grid": grid_obs, "self": self_feat}
        if getattr(self.config, "obs_centralized_critic", False):
            if not hasattr(self, "_cached_global_step") or self._cached_global_step != self.current_step:
                from train.centralized import build_centralized_global_state
                self._cached_global_grid, self._cached_global_scalars = build_centralized_global_state(
                    self, target_h=10, target_w=10
                )
                self._cached_global_step = self.current_step
            obs_dict["global"] = self._cached_global_grid
            obs_dict["global_scalars"] = self._cached_global_scalars

        return obs_dict

    @property
    def verdict(self) -> str:
        """Returns episode outcome: 'WIN', 'LOSE_WIPED', 'LOSE_TIMEOUT', or 'IN_PROGRESS'."""
        all_hostages_rescued = all(h.rescued for h in self.hostages)
        if all_hostages_rescued:
            return "WIN"
        all_rescuers_dead = not any(r.alive for r in self.rescuers.values())
        if all_rescuers_dead:
            return "LOSE_WIPED"
        if getattr(self, "in_overtime", False):
            return "IN_PROGRESS"
        if self.current_step >= self.config.max_steps:
            return "LOSE_TIMEOUT"
        return "IN_PROGRESS"

    def _get_info(self, agent_id: str) -> Dict[str, Any]:
        """Returns diagnostic info for an agent."""
        rescuer = self.rescuers[agent_id]
        rescued_count = sum(1 for h in self.hostages if h.rescued)
        living_count = sum(1 for r in self.rescuers.values() if r.alive)
        return {
            "alive": rescuer.alive,
            "hp": rescuer.hp,
            "max_hp": getattr(rescuer, "max_hp", 2),
            "invulnerable_steps": rescuer.invulnerable_steps,
            "carrying": rescuer.carrying,
            "floor": rescuer.floor,
            "row": rescuer.row,
            "col": rescuer.col,
            "hostages_rescued": rescued_count,
            "total_hostages": self.config.n_hostages,
            "step": self.current_step,
            "max_steps": self.config.max_steps,
            "effective_max_steps": getattr(self, "effective_max_steps", self.config.max_steps),
            "in_overtime": getattr(self, "in_overtime", False),
            "overtime_steps": getattr(self, "overtime_steps", 0),
            "fire_deaths": getattr(self, "fire_deaths", 0),
            "hp_lost": getattr(self, "hp_lost", 0),
            "fire_dashes_survived": getattr(self, "fire_dashes_survived", 0),
            "rescuers_alive": living_count,
            "total_rescuers": self.config.n_rescuers,
            "verdict": self.verdict,
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
            # Search nearest valid EMPTY or ASH neighbor
            neighbors = []
            for dr, dc in (
                (-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (-1, 1), (1, -1), (1, 1),
            ):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.config.grid_h and 0 <= nc < self.config.grid_w:
                    if self.grids[f, nr, nc] in (Tile.EMPTY, Tile.ASH):
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

    def _spread_fire(self) -> bool:
        """Spreads fire using the active pluggable fire controller. Returns True if any new fire spawned."""
        if not self.config.fire_enabled:
            return False
        if self.fire_controller is None:
            return False
        if len(self.fire_positions) >= self.config.fire_max_tiles:
            return False

        new_ignitions = self.fire_controller.step(
            current_step=self.current_step,
            grids=self.grids,
            fire_positions=self.fire_positions,
            rescuers=self.rescuers,
            hostages=self.hostages,
            np_random=self.np_random,
        )
        if not new_ignitions:
            return False

        budget = getattr(self.config, "fire_ignitions_per_step", 1)
        spread_occurred = False
        ignited_count = 0
        for f, r, c in new_ignitions:
            if ignited_count >= budget:
                break
            if (
                0 <= f < self.config.floors
                and 0 <= r < self.config.grid_h
                and 0 <= c < self.config.grid_w
                and self.grids[f, r, c] in (Tile.EMPTY, Tile.ASH)
                and len(self.fire_positions) < self.config.fire_max_tiles
            ):
                # No point-blank check
                if getattr(self.config, "no_point_blank_ignition", True):
                    is_point_blank = False
                    for rescuer in self.rescuers.values():
                        if rescuer.alive and rescuer.floor == f:
                            if abs(rescuer.row - r) + abs(rescuer.col - c) <= 1:
                                is_point_blank = True
                                break
                    if is_point_blank:
                        continue

                # Ash cooldown check
                if self.grids[f, r, c] == Tile.ASH and self.ash_cooldowns.get((f, r, c), 0) > 0:
                    continue

                self.grids[f, r, c] = Tile.FIRE
                pos = (f, r, c)
                self.fire_positions.append(pos)
                self.fire_ages[pos] = 0
                if pos in self.ash_cooldowns:
                    del self.ash_cooldowns[pos]
                spread_occurred = True
                ignited_count += 1

        return spread_occurred

    def _recompute_shaping_maps(self, update_exits: bool = True) -> None:
        """Precomputes or updates multi-floor BFS shortest-path distance maps for potential shaping and observation."""
        if not (self.config.enable_shaping or getattr(self.config, "obs_bfs_features", True)) or not self.building_map:
            return
        if update_exits or not hasattr(self, "_exit_dist_map") or self._exit_dist_map is None:
            self._exit_dist_map, self._exit_origin_floor = self._compute_multi_floor_bfs_dist_map(
                self.building_map.exits, return_origins=True
            )

        hostage_targets = [
            (h.floor, h.row, h.col)
            for h in self.hostages
            if not h.rescued and h.carried_by is None
        ]
        self._hostage_dist_map, self._hostage_origin_floor = self._compute_multi_floor_bfs_dist_map(
            hostage_targets, return_origins=True
        )

    def _compute_multi_floor_bfs_dist_map(
        self, targets: List[Tuple[int, int, int]], return_origins: bool = False
    ) -> Any:
        """Computes shortest path 3D BFS distance map across floors and stairs respecting walls."""
        floors, h, w = self.config.floors, self.config.grid_h, self.config.grid_w
        dist_map = np.full((floors, h, w), fill_value=999, dtype=np.int32)
        origin_floor = np.full((floors, h, w), fill_value=-1, dtype=np.int32)
        if not targets:
            return (dist_map, origin_floor) if return_origins else dist_map

        queue: deque[Tuple[int, int, int]] = deque()
        for tf, tr, tc in targets:
            if 0 <= tf < floors and 0 <= tr < h and 0 <= tc < w:
                dist_map[tf, tr, tc] = 0
                origin_floor[tf, tr, tc] = tf
                queue.append((tf, tr, tc))

        stairs_set = set(self.building_map.stairs) if self.building_map else set()

        while queue:
            f, r, c = queue.popleft()
            d = dist_map[f, r, c]
            orig_f = origin_floor[f, r, c]

            # 1. Lateral moves on same floor (impassable: WALL and FIRE)
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    if self.grids[f, nr, nc] not in (Tile.WALL, Tile.FIRE) and dist_map[f, nr, nc] > d + 1:
                        dist_map[f, nr, nc] = d + 1
                        origin_floor[f, nr, nc] = orig_f
                        queue.append((f, nr, nc))

            # 2. Inter-floor transition via stairs (impassable: WALL and FIRE)
            if (r, c) in stairs_set and self.grids[f, r, c] == Tile.STAIRS:
                for nf in (f - 1, f + 1):
                    if 0 <= nf < floors:
                        if self.grids[nf, r, c] not in (Tile.WALL, Tile.FIRE) and dist_map[nf, r, c] > d + 1:
                            dist_map[nf, r, c] = d + 1
                            origin_floor[nf, r, c] = orig_f
                            queue.append((nf, r, c))

        return (dist_map, origin_floor) if return_origins else dist_map

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

        # Decrement invulnerability steps for living rescuers from prior hits
        for agent in active_stepping_agents:
            rescuer = self.rescuers[agent]
            if rescuer.alive and rescuer.invulnerable_steps > 0:
                rescuer.invulnerable_steps -= 1

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
                if self.config.enable_shaping and hasattr(self, "_exit_dist_map") and self._exit_dist_map is not None:
                    f, r, c = rescuer.floor, rescuer.row, rescuer.col
                    if rescuer.carrying:
                        prev_dist = int(self._exit_dist_map[f, r, c])
                    elif hasattr(self, "_hostage_dist_map") and self._hostage_dist_map is not None:
                        prev_dist = int(self._hostage_dist_map[f, r, c])

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
                        f, r, c = rescuer.floor, rescuer.row, rescuer.col
                        if rescuer.carrying:
                            curr_dist = int(self._exit_dist_map[f, r, c])
                        else:
                            curr_dist = int(self._hostage_dist_map[f, r, c])
                        if curr_dist < 999:
                            rewards[agent] += (prev_dist - curr_dist) * self.config.shaping_coef

                    # Step onto FIRE: costs 1 HP unless invulnerable
                    if self.grids[rescuer.floor, nr, nc] == Tile.FIRE:
                        if rescuer.invulnerable_steps > 0:
                            self.fire_dashes_survived += 1
                        else:
                            rescuer.hp -= 1
                            self.hp_lost += 1
                            rewards[agent] += getattr(self.config, "hit_penalty", -1.0)
                            rescuer.invulnerable_steps = getattr(self.config, "invulnerability_steps", 2)
                            if rescuer.hp <= 0:
                                rescuer.alive = False
                                self.fire_deaths += 1
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
                                    rescuer.hp = min(getattr(self.config, "rescuer_hp", 2), rescuer.hp + 1)
                                    break

            elif action == Action.INTERACT:
                curr_f = rescuer.floor
                curr_r = rescuer.row
                curr_c = rescuer.col
                tile_val = self.grids[curr_f, curr_r, curr_c]

                # A. Use STAIRS -> transition to the other floor (if multi-floor)
                if tile_val == Tile.STAIRS and self.config.floors > 1:
                    prev_dist = None
                    if self.config.enable_shaping and hasattr(self, "_exit_dist_map") and self._exit_dist_map is not None:
                        if rescuer.carrying:
                            prev_dist = int(self._exit_dist_map[curr_f, curr_r, curr_c])
                        elif hasattr(self, "_hostage_dist_map") and self._hostage_dist_map is not None:
                            prev_dist = int(self._hostage_dist_map[curr_f, curr_r, curr_c])

                    target_floor = 1 - curr_f if self.config.floors == 2 else (curr_f + 1) % self.config.floors
                    rescuer.floor = target_floor
                    if rescuer.carrying:
                        for h in self.hostages:
                            if h.carried_by == agent:
                                h.floor = target_floor

                    # Shaping reward and goal-directed transition bonus for vertical stairs
                    if prev_dist is not None and prev_dist < 999:
                        if rescuer.carrying:
                            curr_dist = int(self._exit_dist_map[target_floor, curr_r, curr_c])
                        else:
                            curr_dist = int(self._hostage_dist_map[target_floor, curr_r, curr_c])
                        if curr_dist < 999:
                            rewards[agent] += (prev_dist - curr_dist) * self.config.shaping_coef
                            if curr_dist < prev_dist:
                                rewards[agent] += 0.5  # Stair transition incentive in direction of goal

                    # Cross-floor evacuation transit incentive: carrying hostage down to exit floor (once per hostage)
                    if rescuer.carrying and target_floor == 0 and curr_f > 0:
                        for h in self.hostages:
                            if h.carried_by == agent and not h.evacuated_to_ground:
                                h.evacuated_to_ground = True
                                rewards[agent] += 3.0
                                break

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
                            rescuer.hp = min(getattr(self.config, "rescuer_hp", 2), rescuer.hp + 1)
                            break

        if hostage_state_changed:
            self._recompute_shaping_maps(update_exits=False)

        # Mild anti-clustering incentives:
        # 1. Discourage redundant stacking on same cell
        occupancy: Dict[Tuple[int, int, int], int] = {}
        for r in self.rescuers.values():
            if r.alive:
                pos = (r.floor, r.row, r.col)
                occupancy[pos] = occupancy.get(pos, 0) + 1
        for agent in active_stepping_agents:
            r = self.rescuers[agent]
            if r.alive and occupancy.get((r.floor, r.row, r.col), 0) > 1:
                rewards[agent] -= 0.01

        # 2. Discourage crowding one floor when hostages remain on other floors
        if self.config.floors > 1:
            living_rescuers_per_floor = [0] * self.config.floors
            for r in self.rescuers.values():
                if r.alive:
                    living_rescuers_per_floor[r.floor] += 1
            unrescued_hostages_per_floor = [0] * self.config.floors
            for h in self.hostages:
                if not h.rescued and h.carried_by is None:
                    unrescued_hostages_per_floor[h.floor] += 1

            if len(unrescued_hostages_per_floor) >= 2 and unrescued_hostages_per_floor[0] > 0 and unrescued_hostages_per_floor[1] > 0:
                for agent in active_stepping_agents:
                    r = self.rescuers[agent]
                    if r.alive and living_rescuers_per_floor[r.floor] > 2:
                        rewards[agent] -= 0.01

        # --- 2. Shared Team Completion Bonus ---
        all_hostages_rescued = all(h.rescued for h in self.hostages)
        if new_hostage_delivered and all_hostages_rescued:
            for agent in active_stepping_agents:
                rewards[agent] += self.config.team_completion_reward

        # --- 3. Dynamic Fire Spread, Burnout to ASH & Path Rerouting ---
        # A. Fire Burnout to ASH
        fire_burned_out = False
        expired_fires = []
        for pos in list(self.fire_positions):
            age = self.fire_ages.get(pos, 0) + 1
            self.fire_ages[pos] = age
            if age >= getattr(self.config, "fire_lifetime", 30):
                expired_fires.append(pos)

        for f, r, c in expired_fires:
            pos = (f, r, c)
            if pos in self.fire_positions:
                self.fire_positions.remove(pos)
            if pos in self.fire_ages:
                del self.fire_ages[pos]
            self.grids[f, r, c] = Tile.ASH
            self.ash_cooldowns[pos] = getattr(self.config, "ash_cooldown", 40)
            fire_burned_out = True

        # B. Decrement ash cooldowns
        for pos in list(self.ash_cooldowns.keys()):
            self.ash_cooldowns[pos] -= 1
            if self.ash_cooldowns[pos] <= 0:
                self.ash_cooldowns[pos] = 0

        # C. Fire Spread
        fire_spread_occurred = self._spread_fire()
        if fire_spread_occurred or fire_burned_out:
            self._recompute_shaping_maps(update_exits=True)

        # D. Check if spreading fire engulfed any living rescuers
        for agent in active_stepping_agents:
            rescuer = self.rescuers[agent]
            if rescuer.alive and self.grids[rescuer.floor, rescuer.row, rescuer.col] == Tile.FIRE:
                if rescuer.invulnerable_steps > 0:
                    self.fire_dashes_survived += 1
                else:
                    rescuer.hp -= 1
                    self.hp_lost += 1
                    rewards[agent] += getattr(self.config, "hit_penalty", -1.0)
                    rescuer.invulnerable_steps = getattr(self.config, "invulnerability_steps", 2)
                    if rescuer.hp <= 0:
                        rescuer.alive = False
                        self.fire_deaths += 1
                        rewards[agent] += self.config.death_reward
                        terminations[agent] = True
                        self._drop_carried_hostage_on_death(rescuer)
                        hostage_state_changed = True

        # --- 4. Episode Termination & Truncation Checks ---
        all_rescuers_dead = not any(r.alive for r in self.rescuers.values())

        # Overtime rule: if enabled and step horizon reached, do not truncate while any living rescuer carries
        any_carrying = any(r.alive and r.carrying for r in self.rescuers.values())
        if (
            self.config.carry_overtime
            and (self.current_step >= self.config.max_steps)
            and any_carrying
            and not all_hostages_rescued
            and not all_rescuers_dead
        ):
            self.in_overtime = True
            self.overtime_steps = self.current_step - self.config.max_steps
            while self.current_step >= self.effective_max_steps:
                self.effective_max_steps += 50
            is_truncated = False
        else:
            self.in_overtime = False
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
