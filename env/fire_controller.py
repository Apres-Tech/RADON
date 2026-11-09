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
