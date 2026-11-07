"""Environment factory bridging PettingZoo ParallelEnv to SB3 VecEnv with parameter sharing."""

from typing import Optional
import gymnasium.spaces as spaces
import numpy as np
import supersuit
from supersuit.multiagent_wrappers import black_death
from supersuit.vector.concat_vec_env import ConcatVecEnv

from env.config import EnvConfig
from env.rescue_env import RadonRescueEnv


def _patch_supersuit_compatibility() -> None:
    """Patches SuperSuit's black_death_par and ConcatVecEnv for Gymnasium and Dict observation spaces."""

    def _zero_obs_for_space(space):
        if isinstance(space, spaces.Box):
            return np.zeros_like(space.low)
        elif isinstance(space, spaces.Dict):
            return {k: np.zeros(s.shape, dtype=s.dtype) for k, s in space.spaces.items()}
        raise ValueError(f"Unsupported observation space type: {type(space)}")

    def patched_check_valid(self):
        for agent in self.agents:
            space = self.observation_space(agent)
            assert isinstance(
                space, (spaces.Box, spaces.Dict)
            ), f"observation spaces for black death must be Box or Dict spaces, is {space}"

    def patched_reset(self, seed=None, options=None):
        obss, infos = self.env.reset(seed=seed, options=options)
        self.agents = self.env.agents[:]
        self._check_valid_for_black_death()
        black_obs = {
            agent: _zero_obs_for_space(self.observation_space(agent))
            for agent in self.agents
            if agent not in obss
        }
        return {**obss, **black_obs}, infos

    def patched_step(self, actions):
        active_actions = {agent: actions[agent] for agent in self.env.agents}
        obss, rews, terms, truncs, infos = self.env.step(active_actions)
        black_obs = {
            agent: _zero_obs_for_space(self.observation_space(agent))
            for agent in self.agents
            if agent not in obss
        }
        black_rews = {agent: 0.0 for agent in self.agents if agent not in obss}
        black_infos = {agent: {} for agent in self.agents if agent not in obss}
        terminations = np.fromiter(terms.values(), dtype=bool) if terms else np.array([True])
        truncations = np.fromiter(truncs.values(), dtype=bool) if truncs else np.array([True])
        env_is_done = (terminations | truncations).all()
        total_obs = {**black_obs, **obss}
        total_rews = {**black_rews, **rews}
        total_infos = {**black_infos, **infos}
        total_dones = {agent: env_is_done for agent in self.agents}
        if env_is_done:
            self.agents.clear()
        return total_obs, total_rews, total_dones, total_dones, total_infos

    black_death.black_death_par._check_valid_for_black_death = patched_check_valid
    black_death.black_death_par.reset = patched_reset
    black_death.black_death_par.step = patched_step

    # Compatibility method for SB3 set_random_seed
    if not hasattr(ConcatVecEnv, "seed"):
        ConcatVecEnv.seed = lambda self, seed=None: [seed]


# Apply patch upon module import
_patch_supersuit_compatibility()


def make_rescue_vec_env(
    config: Optional[EnvConfig] = None,
    num_vec_envs: int = 1,
    num_cpus: int = 1,
    seed: Optional[int] = None,
):
    """Creates a vectorized environment for SB3 with parameter sharing across all agents.

    SuperSuit wrappers applied in strict order:
      1. supersuit.black_death_v3(env)
      2. supersuit.pettingzoo_env_to_vec_env_v1(env)
      3. supersuit.concat_vec_envs_v1(env, num_vec_envs, num_cpus, base_class='stable_baselines3')
    """
    cfg = config if config is not None else EnvConfig.training_day2(seed=seed)

    def _env_fn():
        env = RadonRescueEnv(config=cfg)
        # 1. black_death_v3: keeps dead agents active with zero observations until episode end
        env = supersuit.black_death_v3(env)
        # 2. pettingzoo_env_to_vec_env_v1: converts 4 PettingZoo agents into a vectorized environment
        vec_env = supersuit.pettingzoo_env_to_vec_env_v1(env)
        return vec_env

    # 3. concat_vec_envs_v1: concatenates environments into an SB3-compatible VecEnv
    base_vec_env = _env_fn()
    final_vec_env = supersuit.concat_vec_envs_v1(
        base_vec_env,
        num_vec_envs=num_vec_envs,
        num_cpus=num_cpus,
        base_class="stable_baselines3",
    )

    return final_vec_env
