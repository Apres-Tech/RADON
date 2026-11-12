<div align="center">

# 🔥 RADON

**A multi-agent reinforcement learning search-and-rescue arena.**

AI squads learn — from scratch — to evacuate hostages from procedurally generated, multi-floor burning buildings, while a **second AI controls the fire and learns to stop them.**

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-ee4c2c.svg)](https://pytorch.org/)
[![PettingZoo](https://img.shields.io/badge/PettingZoo-1.27-green.svg)](https://pettingzoo.farama.org/)
[![Stable-Baselines3](https://img.shields.io/badge/SB3-2.4.1-orange.svg)](https://stable-baselines3.readthedocs.io/)
[![Three.js](https://img.shields.io/badge/Three.js-3D%20Dashboard-black.svg)](https://threejs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## What is RADON?

RADON is a complete, end-to-end **multi-agent reinforcement learning** system built around a scenario anyone can read at a glance: *a rescue team in a burning building*.

Nothing is scripted. A squad of autonomous agents is given only a reward signal and a world to act in, and learns — over millions of steps — to navigate unfamiliar buildings, split up across floors, carry victims to exits, and dodge spreading fire. Then the fire itself is handed a brain and trained, through **self-play**, to block corridors, cut off stairwells and trap the squad.

Every result in this repository is measured on **held-out maps the agents have never seen**, using a train/test seed split that the training environment can never sample. That discipline was not optional — it caught a bug that had been inflating our own numbers (see [The engineering story](#the-engineering-story)).

An interactive **3D dashboard** lets you watch it live: generate a brand-new building on demand, pick the fire's difficulty, and see the squad win or lose in real time.

---

## Results

All numbers measured on **held-out seeds (50000+)** that the training environment never samples. Fixed horizon, no overtime.

### Rescue performance (no learned adversary)

Showcase 14×14, 2 floors, 4 rescuers, 3 hostages · **n = 100 held-out maps**

| Metric | Random baseline | **RADON** |
|---|---|---|
| Full evacuation (all hostages saved) | 1% | **96.0%** |
| Partial rescue (≥1 saved) | 52% | **100.0%** |
| Mean hostages rescued | 0.76 / 3 | **2.96 / 3** (98.7%) |
| Mean episode length | 219.5 steps | **96.2 steps** |
| Squad survival | 3.35 / 4 | **3.37 / 4** |

### The fair fight (vs. the trained fire adversary)

Large maze 20×20, 2 floors, 5 rescuers, 4 hostages · **n = 25 held-out maps**

| Difficulty preset | Full-win rate | Mean rescued | Fire deaths | Flame dashes survived |
|---|---|---|---|---|
| **Easy** | 84.0% | 3.84 / 4 | 0.00 | 0.12 |
| **Balanced** | **60.0%** | 3.44 / 4 | 1.60 | 4.16 |
| **Brutal** | 32.0% | 3.00 / 4 | 3.24 | 7.64 |

The *Balanced* preset was found by an automated **balance sweep** — a grid search over fire lifetime, ignition budget and spread rate — selecting the configuration where neither side dominates. A fair fight is a tunable, measurable property here, not a vibe.

> **On sample sizes:** the adversarial figures use 25 held-out seeds; a 15-seed diagnostic subset scored 40%. Small samples are noisy — treat the balanced win rate as roughly 40–60% and re-run at `--episodes 100` for a tighter estimate.

### The arms race

Iterated frozen best-response self-play on 20×20 mazes (n = 15 held-out seeds):

| Round | Matchup | Rescuer win rate |
|---|---|---|
| R0 | Rescuer v0 vs. rule-based fire | 46.7% |
| R1 | Rescuer v0 vs. **trained fire v1** | **0.0%** ← the fire learns to trap chokepoints |
| R2 | **Rescuer v2** vs. trained fire v1 | 13.3% ← the squad adapts |
| R3 | Rescuer v2 vs. **trained fire v2** | 33.3% ← the fire adapts back |

---

## Features

- **Procedurally generated buildings** — a fresh multi-floor layout every episode, in two styles: open office floorplans and large labyrinthine mazes with chokepoints and dead ends.
- **Guaranteed solvable** — every generated map is validated by a 3D breadth-first search through stairwells; if any hostage, spawn or staircase can't reach an exit, the map is discarded and regenerated.
- **True multi-agent RL** — PPO with parameter sharing; agents learn cooperation (splitting floors, dividing victims) purely from reward.
- **Egocentric, size-invariant perception** — each agent sees only a 7×7 window around itself, so a single trained policy runs on 10×10, 14×14 and 20×20 maps without retraining.
- **Learned adversarial fire** — a second PPO agent with eight tactical ignition actions (block the nearest rescuer, cut the stairs, seal an exit, surround a victim…), trained via iterated frozen best-response self-play.
- **Balanced combat mechanics** — rescuers have HP and invulnerability frames (so they can *dash through* flame), fire burns out into passable **ash**, and the fire is rate-limited and forbidden from igniting point-blank.
- **Carry overtime** — if a rescuer is carrying a victim when the clock expires, time extends until that rescue resolves (live demo only; benchmarks stay on a fixed horizon).
- **Interactive 3D dashboard** — Three.js multi-floor rendering with smooth interpolated motion, live generation on unseen maps, difficulty presets, HP/ash telemetry, and clear WIN/LOSE verdicts.
- **Honest evaluation by construction** — disjoint train/test seed ranges, held-out benchmarking, failure-mode diagnostics, and curation that is always labelled as curation.

---

## Quick start

```bash
git clone https://github.com/Apres-Tech/RADON.git
cd RADON

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Optional — confirm GPU acceleration:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

Launch the dashboard:

```bash
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000**, pick a layout and a difficulty, and hit **Generate on Unseen Map**. The policy runs live on a building it has never seen.

> CPU-only works fine for inference and the dashboard. Training is much faster on a CUDA GPU.

---

## How it works

### The environment

A [PettingZoo](https://pettingzoo.farama.org/) `ParallelEnv` (verified against the official `parallel_api_test`) in which all agents act simultaneously each tick.

Buildings are tile grids across two floors. Seven tile types: `EMPTY`, `WALL`, `STAIRS`, `EXIT`, `HOSTAGE`, `FIRE`, `ASH`. Rescuers are tracked as separate entities (position, carrying state, HP, alive), cleanly decoupled from the static map. Exits exist only on the ground floor, so upper-floor victims must be carried **down** through stairwells.

### What an agent sees

| Component | Shape | Contents |
|---|---|---|
| Local grid | `(7, 7, 7)` | 7 one-hot channels — walls, fire, hostages, exits, stairs, other rescuers, ash — over a 7×7 egocentric window |
| Self vector | `(14,)` | carrying flag, floor, steps remaining, position, hostages remaining, HP, **+ 7 BFS navigation features** |

The seven BFS features are the key to competence: normalized distance to the current target, a target-floor indicator, a unit direction vector along the shortest corridor, and the step-cost delta for each of the four moves. The target is the nearest unrescued hostage when free, and the nearest exit when carrying.

Because perception is purely local, the policy is **map-size invariant** — the same weights run on any grid dimension.

**Actions** are `Discrete(6)`: four-directional movement, stay, and a context-sensitive `INTERACT` for stairwells. Hostage pickup and delivery happen automatically on contact.

### Rewards

Per-agent rewards for pickups and deliveries, a step penalty for urgency, a death penalty, and a **shared team bonus** when everyone is evacuated — mixing individual drive with collective incentive. On top sits **potential-based BFS shaping**: a dense, policy-invariant "getting warmer" signal proportional to the reduction in true shortest-path distance, computed across floors and treating fire as impassable so routes bend around hazards.

### Training

Stable-Baselines3 **PPO** with parameter sharing, bridged to PettingZoo through SuperSuit. Two integration details worth knowing if you fork this:

- Agents die at different times, which breaks the fixed-agent-count assumption of vectorised training. The `black_death` wrapper handles it — patched here to support `Dict` observation spaces, which it does not do out of the box.
- SB3's default CNN assumes Atari-sized frames and fails on a 7×7 grid, so RADON ships a custom feature extractor (small CNN over the grid channels + MLP over the self vector).

Also included: a **RecurrentPPO (LSTM)** variant and an **asymmetric centralized critic** (MAPPO-style) in which the critic consumes a canonical down-sampled global state while the actor stays strictly local — preserving size invariance.

### The adversary

The fire is a pluggable controller with two implementations: a rule-based escalating curriculum, and a learned PPO policy. The learned fire observes a size-invariant pooled global state and chooses among eight tactical ignitions, constrained to frontier cells (it can only spread next to existing fire, never teleport). Training uses **iterated frozen best-response** — freeze one side, train the other, swap — which keeps each phase ordinary single-policy PPO and avoids the instability of simultaneous co-training.

### The dashboard

A FastAPI backend serves recorded replays and exposes a `/generate` endpoint that runs the trained policy live on a fresh held-out seed. A Three.js front end renders both floors in 3D with interpolated motion, animated fire and ash, per-rescuer HP, difficulty presets, and explicit WIN/LOSE verdicts. Pre-recorded replays remain playable if the backend is offline.

---

## The engineering story

Three findings shaped this project more than any architecture choice. They're documented here because they're the genuinely transferable lessons.

**1. A 100% success rate that was actually 0%.** An early milestone reported near-perfect performance. Evaluating on fresh maps returned *zero*. The cause was a single line: the training entry point pinned `config.seed = 42`, and the generator honoured it on every reset — so across a **1,000,000-step run the agents saw exactly one building.** They had memorised a map, not learned to rescue. The fix was map randomization every episode plus a strict disjoint train/test seed protocol, and every number since has been measured on held-out maps. This is the Procgen/CoinRun overfitting lesson, rediscovered the hard way.

**2. The plateau was an information limit, not an effort limit.** Extending training from 250k to 1.65M steps barely moved the win rate. That flatness was the clue: the agents had a 7×7 window and no memory, so whenever no hostage was in view they had *no information at all* and could only wander — 73.7% of all agent-steps were spent revisiting tiles they'd just left. No amount of extra training fixes a blind agent. Exposing the BFS shortest-path gradient — already computed every step, but only consumed as reward — as an **observation** took the full-win rate from ~6.7% to 96%.

**3. An overpowered opponent isn't a smart opponent.** When the trained fire was switched on, the squad was annihilated — not because the fire was clever, but because its mechanics were unfair: one touch killed, fire spread by area while rescuers moved by line, and burnt tiles stayed blocked forever. The fix was mechanical, not behavioural — HP with invulnerability frames, fire that burns out into passable ash, an ignition rate limit, and no point-blank ignition — followed by retraining both sides against each other and sweeping the parameters until the win rate landed near even.

A corollary worth stating plainly: the BFS navigation feature hands the agents their pathfinding. What remains learned is everything interesting — which rescuer takes which victim, when to dash through flame, when to abandon one person to save another, how to divide floors. We think that's the right trade, but it should be read honestly rather than sold as emergent navigation.

---

## Usage

**Watch an episode in the terminal**

```bash
python -m train.watch --model-path agents/best_bfs_mlp.zip --preset showcase --seed 50042 --delay 0.05
```

Add `--fire-model agents/day10_fire_r2.zip` to watch the squad fight the learned adversary.

**Benchmark on held-out maps**

```bash
python -m train.evaluate \
  --model-path agents/best_bfs_mlp.zip \
  --preset showcase --episodes 100 --base-seed 50000 --fresh-maps
```

**Train**

```bash
# standard PPO
python -m train.train_ppo --preset showcase --total-timesteps 2000000 --num-vec-envs 8

# long, resumable, with held-out tracking
python -m train.train_long --preset maze --total-timesteps 20000000 --resume

# curriculum (grow map size as the policy improves)
python -m train.train_curriculum

# adversarial self-play (iterated frozen best-response)
python -m train.train_day10_selfplay
```

**Diagnose failures and balance the fight**

```bash
python -m train.diagnose --model-path agents/day10_rescuer_r2.zip --preset maze_balanced
python -m train.balance_sweep
```

See [`RUNBOOK.md`](RUNBOOK.md) for the full operator guide, and [`DEVLOG.md`](DEVLOG.md) for the dated build log.

---

## Project structure

```
RADON/
├── env/                    # the simulation
│   ├── rescue_env.py       # PettingZoo ParallelEnv, rewards, BFS shaping, HP, overtime
│   ├── mapgen.py           # procedural generation + 3D BFS solvability guarantee
│   ├── fire_controller.py  # pluggable fire: rule-based curriculum / learned policy
│   ├── fire_env.py         # the adversary's own training environment
│   ├── config.py           # presets: showcase, maze, maze_balanced/easy/brutal, …
│   ├── tiles.py            # tile + action enums, observation channels
│   └── replay.py           # episode recorder (stable JSON schema)
├── train/                  # training, evaluation, diagnostics
│   ├── train_ppo.py  train_long.py  train_curriculum.py  train_day10_selfplay.py
│   ├── extractor.py        # custom feature extractor for the Dict observation
│   ├── centralized.py      # asymmetric (MAPPO-style) centralized critic
│   ├── make_env.py         # PettingZoo → SB3 bridge (black_death, vec envs)
│   ├── evaluate.py  diagnose.py  balance_sweep.py  watch.py
│   └── eval_protocol.py    # disjoint train/test seed ranges
├── server/app.py           # FastAPI backend (replays, metrics, live /generate)
├── dashboard/              # Three.js 3D viewer
├── agents/                 # trained checkpoints
├── metrics/  replays/      # curves, benchmarks, recorded episodes
├── tests/                  # observation shapes, size-invariance, balance mechanics
└── docs/                   # technical writeup + slides
```

---

## Limitations

- **Navigation is given, not discovered.** The BFS gradient in the observation is what broke the performance ceiling; coordination, triage and fire evasion remain learned, but pathfinding does not.
- **Small evaluation samples for the adversarial results.** 25 held-out seeds is noisy; a 15-seed subset of the same configuration scored 40% against 60%. Re-run with more episodes for publication-grade figures.
- **The arms race is asymmetric.** The trained fire degrades the squad far more decisively than the squad recovers; a balanced co-evolutionary equilibrium would need many more alternation rounds.
- **Two floors only**, with modest agent and hostage counts. Nothing fundamental blocks scaling, but it hasn't been validated.
- **The LSTM variant underperformed** the simpler reactive policy here, largely due to ~3.5× lower training throughput rather than a flaw in the idea.

---

## Roadmap

- Larger evaluation sets and confidence intervals on every reported figure
- More self-play rounds toward a genuinely balanced equilibrium
- Degraded/limited-range navigation signal, to make search itself hard again
- Three or more floors, larger squads, victim triage under scarcity
- Communication channels between agents
- Headless benchmark CLI and reproducible result artifacts

---

## Tech stack

Python 3.11 · PyTorch 2.5.1 · Gymnasium 1.0 · PettingZoo 1.27 · SuperSuit 3.11 · Stable-Baselines3 2.4.1 · sb3-contrib 2.4.0 · FastAPI · Three.js · Matplotlib

---

## Credits

Built by **Team Discrete** — Prajjwal Pandey ([@Aionix](https://github.com/aionixOS)), Tanishq Dubey, Deepesh Chakravarti — for **Hack on Holiday 2026**, hosted by the Reinforce Club, Scaler School of Technology, against the *EvoArena — Self-Evolving Multi-Agent Reinforcement Learning Gaming Environment* brief.

Further reading: [`docs/RADON_Technical_Writeup.pdf`](docs/RADON_Technical_Writeup.pdf) · [`DEVLOG.md`](DEVLOG.md) · [`RUNBOOK.md`](RUNBOOK.md)

## License

Released under the MIT License — see [`LICENSE`](LICENSE).

---

<div align="center">
<sub>Every figure in this README was measured on maps the agents had never seen.</sub>
</div>