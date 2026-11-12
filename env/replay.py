"""Episode replay recorder with stable JSON schema for RADON.

Stable Replay JSON Schema (used by downstream visualizers and Day 5 Three.js viewer):
----------------------------------------------------------------------------------
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "RADON_EpisodeReplay",
  "type": "object",
  "properties": {
    "config": {
      "type": "object",
      "properties": {
        "floors": { "type": "integer" },
        "grid_h": { "type": "integer" },
        "grid_w": { "type": "integer" },
        "n_rescuers": { "type": "integer" },
        "n_hostages": { "type": "integer" },
        "view_radius": { "type": "integer" },
        "max_steps": { "type": "integer" },
        "fire_spread_prob": { "type": "number" },
        "carry_capacity": { "type": "integer" },
        "seed": { "type": ["integer", "null"] }
      },
      "required": ["floors", "grid_h", "grid_w", "n_rescuers", "n_hostages", "view_radius", "max_steps"]
    },
    "map": {
      "type": "object",
      "properties": {
        "floors": {
          "type": "array",
          "items": { "type": "array", "items": { "type": "array", "items": { "type": "integer" } } }
        },
        "exits": {
          "type": "array",
          "items": { "type": "array", "items": { "type": "integer" }, "minItems": 3, "maxItems": 3 }
        },
        "stairs": {
          "type": "array",
          "items": { "type": "array", "items": { "type": "integer" }, "minItems": 2, "maxItems": 2 }
        }
      },
      "required": ["floors", "exits", "stairs"]
    },
    "frames": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "t": { "type": "integer" },
          "fire": {
            "type": "array",
            "items": { "type": "array", "items": { "type": "integer" }, "minItems": 3, "maxItems": 3 }
          },
          "hostages": {
            "type": "array",
            "items": {
              "type": "array",
              "items": [{ "type": "integer" }, { "type": "integer" }, { "type": "integer" }, { "type": "boolean" }]
            }
          },
          "rescuers": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "id": { "type": "string" },
                "floor": { "type": "integer" },
                "row": { "type": "integer" },
                "col": { "type": "integer" },
                "carrying": { "type": "boolean" },
                "alive": { "type": "boolean" }
              },
              "required": ["id", "floor", "row", "col", "carrying", "alive"]
            }
          },
          "rewards": {
            "type": "object",
            "additionalProperties": { "type": "number" }
          }
        },
        "required": ["t", "fire", "hostages", "rescuers", "rewards"]
      }
    }
  },
  "required": ["config", "map", "frames"]
}
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np

from env.rescue_env import RadonRescueEnv
from env.tiles import Tile


class EpisodeRecorder:
    """Records simulation trajectories into a stable JSON format for visualization."""

    def __init__(self, env: RadonRescueEnv):
        """Initializes recorder bound to an active RadonRescueEnv instance."""
        self.env = env
        self.config_data: Dict[str, Any] = {
            "floors": env.config.floors,
            "grid_h": env.config.grid_h,
            "grid_w": env.config.grid_w,
            "n_rescuers": env.config.n_rescuers,
            "n_hostages": env.config.n_hostages,
            "view_radius": env.config.view_radius,
            "max_steps": env.config.max_steps,
            "fire_spread_prob": env.config.fire_spread_prob,
            "carry_capacity": env.config.carry_capacity,
            "carry_overtime": getattr(env.config, "carry_overtime", False),
            "seed": env.config.seed,
        }

        # Static map layout representation
        initial_grids = env.building_map.grids.tolist() if env.building_map else []
        exits_list = [list(ex) for ex in env.building_map.exits] if env.building_map else []
        stairs_list = [list(st) for st in env.building_map.stairs] if env.building_map else []

        self.map_data: Dict[str, Any] = {
            "floors": initial_grids,
            "exits": exits_list,
            "stairs": stairs_list,
        }

        self.frames: List[Dict[str, Any]] = []

    def _snapshot_frame(self, t: int, rewards: Dict[str, float]) -> Dict[str, Any]:
        """Captures dynamic state snapshot at time t."""
        # Fires: list of [f, r, c]
        fires = [list(fp) for fp in self.env.fire_positions]

        # Hostages: list of [f, r, c, rescued]
        hostages = [
            [h.floor, h.row, h.col, bool(h.rescued)]
            for h in self.env.hostages
        ]

        # Rescuers: list of rescuer states including HP and invulnerability
        rescuers = [
            {
                "id": r.id,
                "floor": int(r.floor),
                "row": int(r.row),
                "col": int(r.col),
                "carrying": bool(r.carrying),
                "alive": bool(r.alive),
                "hp": int(getattr(r, "hp", 2)),
                "max_hp": int(getattr(r, "max_hp", 2)),
                "invulnerable": bool(getattr(r, "invulnerable_steps", 0) > 0),
            }
            for r in self.env.rescuers.values()
        ]

        # Ash tiles: list of [f, r, c]
        ash_tiles = []
        if hasattr(self.env, "grids"):
            ash_coords = np.argwhere(self.env.grids == Tile.ASH)
            ash_tiles = [[int(f), int(r), int(c)] for f, r, c in ash_coords]

        # Rounded rewards for cleaner JSON
        formatted_rewards = {
            agent_id: round(float(rew), 4)
            for agent_id, rew in rewards.items()
        }

        return {
            "t": t,
            "fire": fires,
            "ash": ash_tiles,
            "hostages": hostages,
            "rescuers": rescuers,
            "rewards": formatted_rewards,
            "in_overtime": getattr(self.env, "in_overtime", False),
            "overtime_steps": getattr(self.env, "overtime_steps", 0),
        }

    def record_initial_frame(self) -> None:
        """Records initial t=0 frame immediately after reset."""
        initial_rewards = {agent: 0.0 for agent in self.env.possible_agents}
        self.frames.append(self._snapshot_frame(t=0, rewards=initial_rewards))

    def record_step(self, rewards: Dict[str, float]) -> None:
        """Appends a frame for the step just taken."""
        self.frames.append(self._snapshot_frame(t=self.env.current_step, rewards=rewards))

    def to_dict(self) -> Dict[str, Any]:
        """Returns the complete replay object adhering to the schema."""
        return {
            "config": self.config_data,
            "map": self.map_data,
            "frames": self.frames,
        }

    def save(self, filepath: Union[str, Path]) -> Path:
        """Saves replay data as structured JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

