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
        fire_spread_prob: Probability of fire spreading to adjacent empty tile per step (default: 0.005).
        fire_spread_start_step: Step threshold before fire begins spreading (default: 10).
        fire_max_tiles: Maximum fire tiles allowed across all floors before spreading caps (default: 40).
        fire_enabled: Master switch for fire spreading dynamics (default: True).
        carry_capacity: Maximum hostages a rescuer can carry simultaneously (default: 1).
        seed: Random seed for procedural generation and dynamics (default: None).

        Reward tunables:
        step_reward: Per-step time penalty (-0.01).
        pickup_reward: Reward on successfully picking up a hostage.
        delivery_reward: Reward on successfully evacuating a hostage at exit.
        death_reward: Penalty if rescuer dies in fire.
        team_completion_reward: Shared team bonus when all hostages are evacuated.
        enable_shaping: Potential-based BFS distance reward shaping for faster convergence.
        shaping_coef: Magnitude of potential-based distance reward.
    """

    floors: int = 2
    grid_h: int = 16
    grid_w: int = 16
    n_rescuers: int = 4
    n_hostages: int = 4
    view_radius: int = 3
    max_steps: int = 200
    fire_spread_prob: float = 0.005
    fire_spread_start_step: int = 10
    fire_max_tiles: int = 40
    fire_enabled: bool = True
    carry_capacity: int = 1
    seed: Optional[int] = None

    # Reward values
    step_reward: float = -0.01
    pickup_reward: float = 2.0
    delivery_reward: float = 10.0
    death_reward: float = -3.0
    team_completion_reward: float = 20.0
    enable_shaping: bool = True
    shaping_coef: float = 0.08

    @property
    def view_size(self) -> int:
        """Egocentric square observation window dimension: 2 * view_radius + 1."""
        return 2 * self.view_radius + 1

    @classmethod
    def training_day3(cls, seed: Optional[int] = None) -> "EnvConfig":
        """Training preset for Day 3: 2 floors, 12x12 grid, 4 rescuers, 4 hostages, max_steps=200."""
        return cls(
            floors=2,
            grid_h=12,
            grid_w=12,
            n_rescuers=4,
            n_hostages=4,
            max_steps=200,
            view_radius=3,
            fire_spread_prob=0.005,
            fire_spread_start_step=15,
            fire_max_tiles=40,
            fire_enabled=True,
            carry_capacity=1,
            seed=seed,
            step_reward=-0.01,
            pickup_reward=5.0,
            delivery_reward=20.0,
            death_reward=-2.0,
            team_completion_reward=30.0,
            enable_shaping=True,
            shaping_coef=0.08,
        )

    @classmethod
    def training_day2(cls, seed: Optional[int] = None) -> "EnvConfig":
        """Tuned training preset for Day 2 / Day 2.5: 1 floor, 12x12 grid, 4 rescuers, 3 hostages, max_steps=120."""
        return cls(
            floors=1,
            grid_h=12,
            grid_w=12,
            n_rescuers=4,
            n_hostages=3,
            max_steps=120,
            view_radius=3,
            fire_spread_prob=0.005,
            fire_spread_start_step=15,
            fire_max_tiles=30,
            fire_enabled=True,
            carry_capacity=1,
            seed=seed,
            step_reward=-0.01,
            pickup_reward=5.0,
            delivery_reward=20.0,
            death_reward=-2.0,
            team_completion_reward=30.0,
            enable_shaping=True,
            shaping_coef=0.08,
        )

    @classmethod
    def sanity(cls, seed: Optional[int] = None) -> "EnvConfig":
        """Sanity curriculum preset for Day 2.5: 1 floor, 8x8 grid, 2 rescuers, 1 hostage, max_steps=60, no fire."""
        return cls(
            floors=1,
            grid_h=8,
            grid_w=8,
            n_rescuers=2,
            n_hostages=1,
            max_steps=60,
            view_radius=3,
            fire_spread_prob=0.0,
            fire_spread_start_step=999,
            fire_max_tiles=0,
            fire_enabled=False,
            carry_capacity=1,
            seed=seed,
            step_reward=-0.01,
            pickup_reward=5.0,
            delivery_reward=20.0,
            death_reward=-2.0,
            team_completion_reward=30.0,
            enable_shaping=True,
            shaping_coef=0.08,
        )
