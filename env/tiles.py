"""Tile definitions, actions, and entity models for RADON."""

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Tuple


class Tile(IntEnum):
    """Grid tile types."""

    EMPTY = 0
    WALL = 1
    STAIRS = 2
    EXIT = 3
    HOSTAGE = 4
    FIRE = 5


class Action(IntEnum):
    """Rescuer discrete action set."""

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    STAY = 4
    INTERACT = 5


# Directional coordinate offsets (d_row, d_col)
ACTION_DELTAS: dict[Action, Tuple[int, int]] = {
    Action.UP: (-1, 0),
    Action.DOWN: (1, 0),
    Action.LEFT: (0, -1),
    Action.RIGHT: (0, 1),
    Action.STAY: (0, 0),
}


# Observation grid channels (6 one-hot layers)
CHANNEL_WALL = 0
CHANNEL_FIRE = 1
CHANNEL_HOSTAGE = 2
CHANNEL_EXIT = 3
CHANNEL_STAIRS = 4
CHANNEL_OTHER_RESCUER = 5
NUM_OBS_CHANNELS = 6


@dataclass
class Rescuer:
    """Rescuer entity model (not a tile)."""

    id: str
    floor: int
    row: int
    col: int
    carrying: bool = False
    alive: bool = True


@dataclass
class Hostage:
    """Hostage entity model."""

    id: int
    floor: int
    row: int
    col: int
    rescued: bool = False
    carried_by: Optional[str] = None

