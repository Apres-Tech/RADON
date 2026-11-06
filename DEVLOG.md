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
