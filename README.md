# RADON — Multi-Agent RL Rescue Simulation

**RADON** is a multi-agent reinforcement learning (MARL) simulation where a squad of autonomous rescuer agents coordinates to evacuate hostages from a burning multi-story building.

## Architecture

```
┌─────────────────────────────────┐
│     PettingZoo ParallelEnv      │
│   (Multi-Floor Procedural Map,  │
│   Fire Dynamics, Rescuers)      │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│    Training & Evaluation (RL)   │
│  (SuperSuit + SB3 PPO / PyTorch)│
└────────────────┬────────────────┘
                 │
                 ▼ (exports stable JSON schema)
┌─────────────────────────────────┐
│     replays/episode_XXX.json    │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│     Three.js Replay Viewer      │
│    (3D Isometric Web Visualizer)│
└─────────────────────────────────┘
```

The system is decoupled into:
1. **Simulation & Environment**: High-performance PettingZoo `ParallelEnv` modeling multi-floor layouts, spreading fires, hostages, and rescuer entities with local partial observability.
2. **Replay Engine**: Zero-dependency structured JSON recorder producing standard trajectories consumed by downstream visualizers.
3. **Visualization & Dashboard**: Lightweight 2D sanity renderer (Day 1) and upcoming 3D Three.js dashboard (Day 5).

## Tech Stack

- **Python**: 3.11
- **Multi-Agent RL**: `PettingZoo` (ParallelEnv API), `SuperSuit`, `Gymnasium`
- **Learning & Inference**: `PyTorch`, `Stable-Baselines3`
- **Numerics & Visualization**: `NumPy`, `Matplotlib`

## Project Structure

```
RADON/
├── env/
│   ├── __init__.py
│   ├── config.py          # EnvConfig dataclass (tunable hyperparameters)
│   ├── tiles.py           # Tile IntEnum & Entity state definitions
│   ├── mapgen.py          # Procedural multi-floor generator + BFS reachability
│   ├── rescue_env.py      # PettingZoo ParallelEnv implementation
│   ├── replay.py          # Episode recorder -> replays/*.json
│   └── demo_random.py     # Random-agent rollout + render sanity check
├── replays/               # Replay JSON recordings and floor visualization PNGs
├── metrics/               # Evaluation metrics and logs
├── agents/                # Saved agent checkpoints and policies
├── requirements.txt       # Pinned dependencies
├── README.md              # Technical overview and documentation
└── DEVLOG.md              # Daily engineering logs and decision records
```

## Setup & Day 1 Quickstart

### 1. Environment Setup

```bash
# Using Python 3.11
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Run Day 1 Sanity Demo

Run a random rollout episode, generate visualization PNGs, export a JSON replay, and verify PettingZoo parallel API compliance:

```bash
python -m env.demo_random
```

This generates:
- Terminal rollout trace and episode statistics
- `replays/day1_floor0.png` and `replays/day1_floor1.png`
- `replays/episode_demo.json`
- PettingZoo `parallel_api_test` verification confirmation
