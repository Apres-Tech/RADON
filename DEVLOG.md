# RADON Development Log

## Day 1 — 2026-11-06 — The Environment
**Built:** repo scaffold; tile/config/entity model; procedural 2-floor map generator; PettingZoo ParallelEnv (6 obs channels, 6 actions); replay recorder + JSON schema; random rollout + render check.
**Decisions:** medium scale (2×16×16, 4 rescuers, 4 hostages); local 7×7 vision; single-floor training first (Day 2), both floors on Day 3.
**Problems & fixes:**
- PettingZoo ParallelEnv compliance: required observation/info/reward dictionaries to contain entries for all agents stepping in the current cycle, while pruning terminated/truncated agents from `self.agents` immediately after checking lifecycle state.
- Space caching: cached `observation_space` and `action_space` using `@lru_cache(maxsize=None)` to guarantee identical space object identity across calls as required by `parallel_api_test`.
- Procedural generation reachability: implemented 3D BFS across floors via shared stairwell coordinates ensuring all hostages, stairs, and rescuer spawns have an unobstructed route to ground floor exits.
**Proof:** parallel_api_test passing; replay JSON + floor PNGs committed.
**Tomorrow (Day 2):** finalize reward, add PPO (SB3 + SuperSuit param-sharing), train one rescuer to evacuate on a single floor — first learning curve.

## Day 2 — 2026-11-07 — First Learning
**Built:** fixed dropped-hostage bug; tuned fire (p=0.005, start@10, cap 40); day2 training config (1 floor, 12x12, 4 rescuers, 3 hostages); PPO pipeline (black_death vec env + custom 7x7 extractor + param sharing); metrics callback.
**Training run:** 150,000 steps, 9.97 minutes, final success rate 0.0% (all 3 evacuated), mean hostages rescued 0.24/3 (random baseline: 0.10/3).
**Decisions:** headline metric = hostages-rescued + success rate (reward noisy with 4 shared agents).
**Problems & fixes:**
- Pre-flight dependencies: CPU PyTorch replaced with `torch==2.5.1+cu121` matching the NVIDIA GeForce RTX 3050 Laptop GPU, with compatible versions `stable-baselines3==2.4.1` and `gymnasium==1.0.0` pinned in requirements.txt.
- SuperSuit black_death Dict space support: SuperSuit's `black_death_par` asserted `Box` observation spaces; patched it to generate zero-dictionaries for Dict observation spaces (`grid` and `self`) so dead agents maintain observation shapes without crashing vectorization.
- SuperSuit ConcatVecEnv seed compatibility: added `.seed()` fallback on `ConcatVecEnv` to satisfy SB3's initialization sequence.
- 7x7 CNN architecture: default NatureCNN requires minimum 36x36 input; designed `RescueExtractor` with 2-layer Conv2d + MLP concatenation (features_dim=160).
- Reward tuning: initial runs with harsh death penalty (-5.0) and small pickup (+1.0) led policy to freeze safely near spawn. Rebalanced pickup (+5.0), delivery (+20.0), team completion (+30.0), softened death (-2.0), added potential distance shaping (coef=0.25), and increased entropy coef (0.04), allowing the trained policy to measurably beat random (0.24 vs 0.10 mean rescued).
**Proof:** metrics/day2_learning_curve.png, replays/day2_trained.json, eval table.
**Tomorrow (Day 3):** scale to 2 floors + stairs, cooperation incentives, dynamic fire during episodes, export multi-floor team replays.
