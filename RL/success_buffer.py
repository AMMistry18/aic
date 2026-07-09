"""Success-biased replay for SAC (SIL / DDPGfD-style exploitation).

Motivation (see RL/CURRICULUM_ENTROPY_TUNING.md §1.3): with a 20k-transition
main ring buffer, transitions from successful episodes age out ~20k env steps
after they happen — right when the curriculum has moved on and needs them
most. Following Vecerik et al. (DDPGfD, arXiv:1707.08817 — real insertion
tasks keep valuable transitions preferentially sampled) and Oh et al.
(Self-Imitation Learning, arXiv:1806.05635 — exploiting past good
trajectories drives deeper exploration), this buffer:

  * caches each env's in-flight episode; when the episode terminates with
    `info["term_status"] == "success"`, its transitions are copied into a
    persistent side ring (`success_buffer_size` transitions, FIFO);
  * serves minibatches as `(1 - success_mix)` uniform samples from the main
    ring + `success_mix` uniform samples from the success ring (once the ring
    holds at least `success_min_fill` transitions).

The success ring is pickled with the buffer (survives checkpoint/resume via
`save_replay_buffer`/`load_replay_buffer`).
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer
from stable_baselines3.common.type_aliases import (
    DictReplayBufferSamples, ReplayBufferSamples)
from stable_baselines3.common.vec_env import VecNormalize


class SuccessMixDictReplayBuffer(DictReplayBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        device: th.device | str = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        success_mix: float = 0.25,
        success_buffer_size: int = 4000,
        success_min_fill: int = 256,
        episode_cache_max: int = 400,
    ):
        super().__init__(
            buffer_size, observation_space, action_space, device=device,
            n_envs=n_envs, optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=handle_timeout_termination)
        self.success_mix = float(np.clip(success_mix, 0.0, 0.9))
        self.success_capacity = max(int(success_buffer_size), 1)
        self.success_min_fill = int(success_min_fill)
        self.episode_cache_max = int(episode_cache_max)

        cap = self.success_capacity
        self._suc_obs = {k: np.zeros((cap, *sp.shape), dtype=sp.dtype)
                         for k, sp in observation_space.spaces.items()}
        self._suc_next_obs = {k: np.zeros((cap, *sp.shape), dtype=sp.dtype)
                              for k, sp in observation_space.spaces.items()}
        self._suc_actions = np.zeros((cap, self.action_dim), dtype=np.float32)
        self._suc_rewards = np.zeros((cap,), dtype=np.float32)
        self._suc_dones = np.zeros((cap,), dtype=np.float32)
        self._suc_pos = 0
        self._suc_size = 0
        # per-env in-flight episode caches (lists of transition tuples)
        self._ep_cache: list[list[tuple]] = [[] for _ in range(n_envs)]

    # ------------------------------------------------------------------ #
    def add(self, obs, next_obs, action, reward, done, infos) -> None:
        super().add(obs, next_obs, action, reward, done, infos)
        action = action.reshape((self.n_envs, self.action_dim))
        for i in range(self.n_envs):
            cache = self._ep_cache[i]
            if len(cache) < self.episode_cache_max:
                timeout = bool(infos[i].get("TimeLimit.truncated", False))
                cache.append((
                    {k: np.array(obs[k][i], copy=True) for k in self._suc_obs},
                    {k: np.array(next_obs[k][i], copy=True) for k in self._suc_next_obs},
                    np.array(action[i], copy=True),
                    float(np.asarray(reward).reshape(-1)[i]),
                    float(bool(done[i]) and not timeout),
                ))
            if bool(done[i]):
                if str(infos[i].get("term_status", "")) == "success":
                    self._flush_success(cache)
                self._ep_cache[i] = []

    def _flush_success(self, cache: list[tuple]) -> None:
        for o, no, a, r, d in cache:
            p = self._suc_pos
            for k in self._suc_obs:
                self._suc_obs[k][p] = o[k]
                self._suc_next_obs[k][p] = no[k]
            self._suc_actions[p] = a
            self._suc_rewards[p] = r
            self._suc_dones[p] = d
            self._suc_pos = (p + 1) % self.success_capacity
            self._suc_size = min(self._suc_size + 1, self.success_capacity)

    # ------------------------------------------------------------------ #
    def sample(self, batch_size: int, env: Optional[VecNormalize] = None
               ) -> DictReplayBufferSamples:
        k = 0
        if self.success_mix > 0.0 and self._suc_size >= self.success_min_fill:
            k = int(round(batch_size * self.success_mix))
            k = min(k, batch_size - 1)
        base = super().sample(batch_size - k, env)
        if k <= 0:
            return base
        idx = np.random.randint(0, self._suc_size, size=k)
        obs_ = self._normalize_obs({key: arr[idx] for key, arr in self._suc_obs.items()}, env)
        nobs_ = self._normalize_obs({key: arr[idx] for key, arr in self._suc_next_obs.items()}, env)
        suc = DictReplayBufferSamples(
            observations={key: self.to_torch(v) for key, v in obs_.items()},
            actions=self.to_torch(self._suc_actions[idx]),
            next_observations={key: self.to_torch(v) for key, v in nobs_.items()},
            dones=self.to_torch(self._suc_dones[idx].reshape(-1, 1)),
            rewards=self.to_torch(self._normalize_reward(
                self._suc_rewards[idx].reshape(-1, 1), env)),
        )
        return DictReplayBufferSamples(
            observations={key: th.cat([base.observations[key], suc.observations[key]])
                          for key in base.observations},
            actions=th.cat([base.actions, suc.actions]),
            next_observations={key: th.cat([base.next_observations[key],
                                            suc.next_observations[key]])
                               for key in base.next_observations},
            dones=th.cat([base.dones, suc.dones]),
            rewards=th.cat([base.rewards, suc.rewards]),
        )

    def success_fill(self) -> int:
        return int(self._suc_size)


class SuccessMixReplayBuffer(ReplayBuffer):
    """Flat-Box observation twin of SuccessMixDictReplayBuffer (same policy),
    for the privileged-state residual training path."""

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: th.device | str = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        success_mix: float = 0.25,
        success_buffer_size: int = 4000,
        success_min_fill: int = 256,
        episode_cache_max: int = 400,
    ):
        super().__init__(
            buffer_size, observation_space, action_space, device=device,
            n_envs=n_envs, optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=handle_timeout_termination)
        self.success_mix = float(np.clip(success_mix, 0.0, 0.9))
        self.success_capacity = max(int(success_buffer_size), 1)
        self.success_min_fill = int(success_min_fill)
        self.episode_cache_max = int(episode_cache_max)

        cap = self.success_capacity
        self._suc_obs = np.zeros((cap, *self.obs_shape), dtype=observation_space.dtype)
        self._suc_next_obs = np.zeros((cap, *self.obs_shape), dtype=observation_space.dtype)
        self._suc_actions = np.zeros((cap, self.action_dim), dtype=np.float32)
        self._suc_rewards = np.zeros((cap,), dtype=np.float32)
        self._suc_dones = np.zeros((cap,), dtype=np.float32)
        self._suc_pos = 0
        self._suc_size = 0
        self._ep_cache: list[list[tuple]] = [[] for _ in range(n_envs)]

    # ------------------------------------------------------------------ #
    def add(self, obs, next_obs, action, reward, done, infos) -> None:
        super().add(obs, next_obs, action, reward, done, infos)
        obs = np.asarray(obs).reshape((self.n_envs, *self.obs_shape))
        next_obs = np.asarray(next_obs).reshape((self.n_envs, *self.obs_shape))
        action = np.asarray(action).reshape((self.n_envs, self.action_dim))
        for i in range(self.n_envs):
            cache = self._ep_cache[i]
            if len(cache) < self.episode_cache_max:
                timeout = bool(infos[i].get("TimeLimit.truncated", False))
                cache.append((
                    np.array(obs[i], copy=True),
                    np.array(next_obs[i], copy=True),
                    np.array(action[i], copy=True),
                    float(np.asarray(reward).reshape(-1)[i]),
                    float(bool(done[i]) and not timeout),
                ))
            if bool(done[i]):
                if str(infos[i].get("term_status", "")) == "success":
                    self._flush_success(cache)
                self._ep_cache[i] = []

    def _flush_success(self, cache: list[tuple]) -> None:
        for o, no, a, r, d in cache:
            p = self._suc_pos
            self._suc_obs[p] = o
            self._suc_next_obs[p] = no
            self._suc_actions[p] = a
            self._suc_rewards[p] = r
            self._suc_dones[p] = d
            self._suc_pos = (p + 1) % self.success_capacity
            self._suc_size = min(self._suc_size + 1, self.success_capacity)

    # ------------------------------------------------------------------ #
    def sample(self, batch_size: int, env: Optional[VecNormalize] = None
               ) -> ReplayBufferSamples:
        k = 0
        if self.success_mix > 0.0 and self._suc_size >= self.success_min_fill:
            k = int(round(batch_size * self.success_mix))
            k = min(k, batch_size - 1)
        base = super().sample(batch_size - k, env)
        if k <= 0:
            return base
        idx = np.random.randint(0, self._suc_size, size=k)
        obs_ = self._normalize_obs(self._suc_obs[idx], env)
        nobs_ = self._normalize_obs(self._suc_next_obs[idx], env)
        return ReplayBufferSamples(
            observations=th.cat([base.observations, self.to_torch(obs_)]),
            actions=th.cat([base.actions, self.to_torch(self._suc_actions[idx])]),
            next_observations=th.cat([base.next_observations, self.to_torch(nobs_)]),
            dones=th.cat([base.dones,
                          self.to_torch(self._suc_dones[idx].reshape(-1, 1))]),
            rewards=th.cat([base.rewards, self.to_torch(self._normalize_reward(
                self._suc_rewards[idx].reshape(-1, 1), env))]),
        )

    def success_fill(self) -> int:
        return int(self._suc_size)


__all__ = ["SuccessMixDictReplayBuffer", "SuccessMixReplayBuffer"]
