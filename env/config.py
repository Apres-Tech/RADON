"""Configuration dataclass for RADON rescue environment."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class EnvConfig:
    """Environment configuration hyperparameters and defaults.

    Tunables:
        floors: Number of building floors (default: 2).
        grid_h: Height of each floor grid (default: 16).
        grid_w: Width of each floor grid (default: 16).
        n_rescuers: Number of rescuer agents (default: 4).
        n_hostages: Number of hostages to evacuate (default: 4).
        view_radius: Radius for egocentric local vision window (default: 3 -> 7x7).
        max_steps: Maximum step horizon per episode (default: 200).
        fire_spread_prob: Probability of fire spreading to adjacent non-wall tile per step (default: 0.02).
        carry_capacity: Maximum hostages a rescuer can carry simultaneously (default: 1).
        seed: Random seed for procedural generation and dynamics (default: None).

        Reward tunables (First pass - marked for Day 2 tuning):
        step_reward: Per-step time penalty (-0.01).
        pickup_reward: Reward on successfully picking up a hostage (+1.0).
        delivery_reward: Reward on successfully evacuating a hostage at exit (+5.0).
        death_reward: Penalty if rescuer dies in fire (-5.0).
        team_completion_reward: Shared team bonus when all hostages are evacuated (+10.0).
    """

    floors: int = 2
    grid_h: int = 16
    grid_w: int = 16
    n_rescuers: int = 4
    n_hostages: int = 4
    view_radius: int = 3
    max_steps: int = 200
    fire_spread_prob: float = 0.02
    carry_capacity: int = 1
    seed: Optional[int] = None

    # First-pass reward values (will be tuned Day 2)
    step_reward: float = -0.01
    pickup_reward: float = 1.0
    delivery_reward: float = 5.0
    death_reward: float = -5.0
    team_completion_reward: float = 10.0

    @property
    def view_size(self) -> int:
        """Egocentric square observation window dimension: 2 * view_radius + 1."""
        return 2 * self.view_radius + 1

