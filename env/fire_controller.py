"""Pluggable Fire Controller interface and escalating curriculum fire implementation for RADON."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Set
import numpy as np

from env.tiles import Tile, Rescuer, Hostage


def get_valid_fire_frontier(
    grids: np.ndarray,
    fire_positions: List[Tuple[int, int, int]],
    grid_h: int,
    grid_w: int,
) -> List[Tuple[int, int, int]]:
    """Identifies all valid frontier cells eligible for ignition.

    Physical safety constraints:
      - Must be orthogonally adjacent to an existing FIRE tile.
      - Must be an EMPTY tile (strictly never WALL, FIRE, EXIT, STAIRS, or HOSTAGE).
    """
    frontier_set: Set[Tuple[int, int, int]] = set()
    for ff, fr, fc in fire_positions:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nfr, nfc = fr + dr, fc + dc
            if 0 <= nfr < grid_h and 0 <= nfc < grid_w:
                if grids[ff, nfr, nfc] == Tile.EMPTY:
                    frontier_set.add((ff, nfr, nfc))
    return sorted(list(frontier_set))


class FireController(ABC):
    """Abstract base class for fire dynamics and adversary ignition controllers."""

    @abstractmethod
    def reset(
        self,
        config,
        grids: np.ndarray,
        fire_positions: List[Tuple[int, int, int]],
        np_random: np.random.Generator,
    ) -> None:
        """Resets internal controller state at the start of an episode."""
        pass

    @abstractmethod
    def step(
        self,
        current_step: int,
        grids: np.ndarray,
        fire_positions: List[Tuple[int, int, int]],
        rescuers: Dict[str, Rescuer],
        hostages: List[Hostage],
        np_random: np.random.Generator,
    ) -> List[Tuple[int, int, int]]:
        """Returns a list of newly ignited cell coordinates [(f, r, c), ...] for this step."""
        pass


class CurriculumFireController(FireController):
    """Rule-based escalating fire controller.

    Agility and spread aggressiveness smoothly ramp up as the episode progresses
    or across training difficulty levels. Guaranteed demo-safe and physically fair.
    """

    def __init__(
        self,
        base_spread_prob: float = 0.008,
        max_spread_prob: float = 0.025,
        escalation_horizon: int = 120,
        start_step: int = 10,
        max_fire_tiles: int = 35,
        max_budget_per_step: int = 2,
    ):
        self.base_spread_prob = base_spread_prob
        self.max_spread_prob = max_spread_prob
        self.escalation_horizon = escalation_horizon
        self.start_step = start_step
        self.max_fire_tiles = max_fire_tiles
        self.max_budget_per_step = max_budget_per_step
        self.config = None

    def reset(
        self,
        config,
        grids: np.ndarray,
        fire_positions: List[Tuple[int, int, int]],
        np_random: np.random.Generator,
    ) -> None:
        self.config = config
        self.start_step = getattr(config, "fire_spread_start_step", self.start_step)
        self.max_fire_tiles = getattr(config, "fire_max_tiles", self.max_fire_tiles)
        self.base_spread_prob = getattr(config, "fire_spread_prob", self.base_spread_prob)

    def get_current_spread_prob(self, current_step: int) -> float:
        """Calculates time-escalated spread probability."""
        if current_step < self.start_step:
            return 0.0
        progress = min(1.0, max(0.0, float(current_step - self.start_step) / float(self.escalation_horizon)))
        return self.base_spread_prob + progress * (self.max_spread_prob - self.base_spread_prob)

    def step(
        self,
        current_step: int,
        grids: np.ndarray,
        fire_positions: List[Tuple[int, int, int]],
        rescuers: Dict[str, Rescuer],
        hostages: List[Hostage],
        np_random: np.random.Generator,
    ) -> List[Tuple[int, int, int]]:
        if not getattr(self.config, "fire_enabled", True):
            return []
        if current_step < self.start_step:
            return []
        if len(fire_positions) >= self.max_fire_tiles:
            return []

        prob = self.get_current_spread_prob(current_step)
        if np_random.random() >= prob:
            return []

        # Find eligible frontier
        frontier = get_valid_fire_frontier(grids, fire_positions, self.config.grid_h, self.config.grid_w)
        if not frontier:
            return []

        # Determine step budget: 1 early, up to 2 later in episode
        remaining_capacity = self.max_fire_tiles - len(fire_positions)
        step_budget = min(
            remaining_capacity,
            self.max_budget_per_step if current_step >= (self.start_step + self.escalation_horizon // 2) else 1,
            len(frontier),
        )

        chosen_indices = np_random.choice(len(frontier), size=step_budget, replace=False)
        return [frontier[i] for i in chosen_indices]


def build_fire_observation(
    grids: np.ndarray,
    fire_positions: List[Tuple[int, int, int]],
    rescuers: Dict[str, Rescuer],
    hostages: List[Hostage],
    current_step: int,
    config,
    frontier: List[Tuple[int, int, int]],
) -> Dict[str, np.ndarray]:
    """Constructs multi-channel global grid and scalar vector for the adversary fire policy.

    Grid channels across all floors (shape: (6 * floors, grid_h, grid_w)):
      Channel 0: Wall
      Channel 1: Fire
      Channel 2: Living rescuers
      Channel 3: Unrescued hostages
      Channel 4: Stairs / Exits
      Channel 5: Eligible fire frontier
    """
    floors = config.floors
    h, w = config.grid_h, config.grid_w
    channels_per_floor = 6
    grid_obs = np.zeros((channels_per_floor * floors, h, w), dtype=np.float32)

    frontier_set = set(frontier)

    for f in range(floors):
        offset = f * channels_per_floor
        # Channel 0: Wall
        grid_obs[offset + 0] = (grids[f] == Tile.WALL).astype(np.float32)
        # Channel 1: Fire
        grid_obs[offset + 1] = (grids[f] == Tile.FIRE).astype(np.float32)
        # Channel 4: Stairs and Exits
        grid_obs[offset + 4] = (grids[f] == Tile.STAIRS).astype(np.float32)
        if f == 0:
            grid_obs[offset + 4] += (grids[f] == Tile.EXIT).astype(np.float32)

    # Channel 2: Rescuers
    for r in rescuers.values():
        if r.alive and 0 <= r.floor < floors and 0 <= r.row < h and 0 <= r.col < w:
            grid_obs[r.floor * channels_per_floor + 2, r.row, r.col] = 1.0

    # Channel 3: Hostages
    for h_obj in hostages:
        if not h_obj.rescued and 0 <= h_obj.floor < floors and 0 <= h_obj.row < h and 0 <= h_obj.col < w:
            grid_obs[h_obj.floor * channels_per_floor + 3, h_obj.row, h_obj.col] = 1.0

    # Channel 5: Frontier
    for ff, fr, fc in frontier_set:
        if 0 <= ff < floors and 0 <= fr < h and 0 <= fc < w:
            grid_obs[ff * channels_per_floor + 5, fr, fc] = 1.0

    # Scalar features (length 6)
    unrescued = sum(1 for ho in hostages if not ho.rescued)
    living = sum(1 for ro in rescuers.values() if ro.alive)
    step_norm = float(current_step) / float(max(1, config.max_steps))
    hostage_norm = float(unrescued) / float(max(1, config.n_hostages))
    rescuer_norm = float(living) / float(max(1, config.n_rescuers))
    max_fires = getattr(config, "fire_max_tiles", 30)
    fire_norm = float(len(fire_positions)) / float(max(1, max_fires))
    frontier_norm = min(1.0, float(len(frontier)) / 25.0)
    f1_hostages = sum(1 for ho in hostages if not ho.rescued and ho.floor == 1)
    f1_ratio = float(f1_hostages) / float(max(1, unrescued))

    scalars = np.array(
        [step_norm, hostage_norm, rescuer_norm, fire_norm, frontier_norm, f1_ratio],
        dtype=np.float32,
    )

    return {"grid": grid_obs, "scalars": scalars}


def select_frontier_ignition(
    action: int,
    frontier: List[Tuple[int, int, int]],
    rescuers: Dict[str, Rescuer],
    hostages: List[Hostage],
    config,
    budget: int = 1,
) -> List[Tuple[int, int, int]]:
    """Maps a discrete action to 1-2 physically valid frontier ignitions.

    Action semantics:
      0: Closest frontier cell to nearest living rescuer (cut off movement)
      1: 2nd closest frontier cell to rescuer (flank / trap)
      2: Closest frontier cell to nearest stairs (block inter-floor transit)
      3: Closest frontier cell to ground-floor exit (block delivery)
      4: Closest frontier cell to nearest unrescued hostage (surround victim)
      5: Center-most frontier cell (deep building expansion)
      6: Random valid frontier cell (exploration)
      7: Hold / Pass (save budget, 0 ignitions)
    """
    if not frontier or action == 7:
        return []

    living_rescuers = [r for r in rescuers.values() if r.alive]
    unrescued_hostages = [h for h in hostages if not h.rescued]

    # Action 0: Closest to nearest rescuer
    if action == 0 and living_rescuers:
        sorted_frontier = sorted(
            frontier,
            key=lambda c: min(abs(c[1] - r.row) + abs(c[2] - r.col) + (0 if c[0] == r.floor else 6) for r in living_rescuers),
        )
        return sorted_frontier[:budget]

    # Action 1: 2nd closest to rescuer
    elif action == 1 and living_rescuers and len(frontier) > 1:
        sorted_frontier = sorted(
            frontier,
            key=lambda c: min(abs(c[1] - r.row) + abs(c[2] - r.col) + (0 if c[0] == r.floor else 6) for r in living_rescuers),
        )
        return sorted_frontier[1 : 1 + budget]

    # Action 2: Closest to stairs (if any)
    elif action == 2:
        stairs_locs = []
        for r in range(config.grid_h):
            for c in range(config.grid_w):
                if hasattr(config, "stairs") and (r, c) in getattr(config, "stairs", []):
                    stairs_locs.append((r, c))
        if stairs_locs:
            sorted_frontier = sorted(
                frontier,
                key=lambda c: min(abs(c[1] - sr) + abs(c[2] - sc) for sr, sc in stairs_locs),
            )
            return sorted_frontier[:budget]
        else:
            return [frontier[0]]

    # Action 3: Closest to exits (floor 0)
    elif action == 3:
        f0_frontier = [c for c in frontier if c[0] == 0]
        chosen_pool = f0_frontier if f0_frontier else frontier
        sorted_frontier = sorted(
            chosen_pool,
            key=lambda c: min(c[1], c[2], config.grid_h - 1 - c[1], config.grid_w - 1 - c[2]),
        )
        return sorted_frontier[:budget]

    # Action 4: Closest to nearest unrescued hostage
    elif action == 4 and unrescued_hostages:
        sorted_frontier = sorted(
            frontier,
            key=lambda c: min(abs(c[1] - h.row) + abs(c[2] - h.col) + (0 if c[0] == h.floor else 6) for h in unrescued_hostages),
        )
        return sorted_frontier[:budget]

    # Action 5: Deepest / center-most
    elif action == 5:
        center_r, center_c = config.grid_h / 2.0, config.grid_w / 2.0
        sorted_frontier = sorted(
            frontier,
            key=lambda c: abs(c[1] - center_r) + abs(c[2] - center_c),
        )
        return sorted_frontier[:budget]

    # Action 6 / Fallback: Deterministic index modulo frontier size
    else:
        idx = action % len(frontier)
        return [frontier[idx]]


class PolicyFireController(FireController):
    """Learned policy-driven fire controller powered by a trained adversary PPO model."""

    def __init__(self, model_path: Optional[str] = None, model: Optional[object] = None, deterministic: bool = False):
        self.model_path = model_path
        self.deterministic = deterministic
        self.model = model
        self.config = None
        if self.model is None and model_path is not None:
            from pathlib import Path
            p = Path(model_path)
            if p.exists():
                from stable_baselines3 import PPO
                self.model = PPO.load(str(p))

    def reset(
        self,
        config,
        grids: np.ndarray,
        fire_positions: List[Tuple[int, int, int]],
        np_random: np.random.Generator,
    ) -> None:
        self.config = config

    def step(
        self,
        current_step: int,
        grids: np.ndarray,
        fire_positions: List[Tuple[int, int, int]],
        rescuers: Dict[str, Rescuer],
        hostages: List[Hostage],
        np_random: np.random.Generator,
    ) -> List[Tuple[int, int, int]]:
        if not getattr(self.config, "fire_enabled", True):
            return []
        start_step = getattr(self.config, "fire_spread_start_step", 10)
        if current_step < start_step:
            return []
        max_tiles = getattr(self.config, "fire_max_tiles", 30)
        if len(fire_positions) >= max_tiles:
            return []

        frontier = get_valid_fire_frontier(grids, fire_positions, self.config.grid_h, self.config.grid_w)
        if not frontier:
            return []

        if self.model is not None:
            obs = build_fire_observation(grids, fire_positions, rescuers, hostages, current_step, self.config, frontier)
            # Expand batch dimension for SB3
            batch_obs = {k: np.expand_dims(v, axis=0) for k, v in obs.items()}
            action, _ = self.model.predict(batch_obs, deterministic=self.deterministic)
            action_idx = int(action[0])
        else:
            action_idx = int(np_random.integers(0, 7))

        return select_frontier_ignition(action_idx, frontier, rescuers, hostages, self.config, budget=1)
