# RADON Development Log

## Day 1 — 2026-11-06 — The Environment
**Built:** repo scaffold; tile/config/entity model; procedural 2-floor map generator; PettingZoo ParallelEnv (6 obs channels, 6 actions); replay recorder + JSON schema; random rollout + render check.
**Decisions:** medium scale (2×16×16, 4 rescuers, 4 hostages); local 7×7 vision; single-floor training first (Day 2), both floors on Day 3.
**Problems & fixes:** Scaffolding virtualenv with Python 3.11 and pinning dependency versions across gymnasium, pettingzoo, supersuit, and stable-baselines3.
**Proof:** parallel_api_test passing; replay JSON + floor PNGs committed.
**Tomorrow (Day 2):** finalize reward, add PPO (SB3 + SuperSuit param-sharing), train one rescuer to evacuate on a single floor — first learning curve.
