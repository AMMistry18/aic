"""Asymmetric SAC and RLPD replay for Student v3."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

import gymnasium as gym
import numpy as np
import torch as th
from torch import nn
import torch.nn.functional as F

from stable_baselines3 import SAC
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import DictReplayBufferSamples
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.sac.policies import Actor, ContinuousCritic, SACPolicy


class ActorHistoryExtractor(BaseFeaturesExtractor):
    """Actor-visible extractor; intentionally never reads privileged state."""

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        actor_dim = int(np.prod(observation_space["actor"].shape))
        self.net = nn.Sequential(
            nn.Flatten(), nn.LayerNorm(actor_dim),
            nn.Linear(actor_dim, 256), nn.SiLU(),
            nn.Linear(256, features_dim), nn.SiLU(),
        )

    def forward(self, observations: dict[str, th.Tensor]) -> th.Tensor:
        return self.net(observations["actor"])


class PrivilegedCriticExtractor(BaseFeaturesExtractor):
    """Critic extractor over actor history plus simulator-only state."""

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        actor_dim = int(np.prod(observation_space["actor"].shape))
        privileged_dim = int(np.prod(observation_space["privileged"].shape))
        total = actor_dim + privileged_dim
        self.net = nn.Sequential(
            nn.LayerNorm(total),
            nn.Linear(total, 256), nn.SiLU(),
            nn.Linear(256, features_dim), nn.SiLU(),
        )

    def forward(self, observations: dict[str, th.Tensor]) -> th.Tensor:
        actor = th.flatten(observations["actor"], start_dim=1)
        privileged = th.flatten(observations["privileged"], start_dim=1)
        return self.net(th.cat([actor, privileged], dim=1))


class AsymmetricSACPolicy(SACPolicy):
    """SAC policy whose actor cannot access privileged observations."""

    def make_actor(self, features_extractor=None) -> Actor:
        extractor = ActorHistoryExtractor(self.observation_space, features_dim=256)
        kwargs = self._update_features_extractor(self.actor_kwargs, extractor)
        actor = Actor(**kwargs).to(self.device)
        # Exact zero residual at initialization: deterministic action = tanh(0).
        mu_layer = actor.mu[0] if isinstance(actor.mu, nn.Sequential) else actor.mu
        nn.init.zeros_(mu_layer.weight)
        nn.init.zeros_(mu_layer.bias)
        if isinstance(actor.log_std, nn.Linear):
            nn.init.zeros_(actor.log_std.weight)
            nn.init.constant_(actor.log_std.bias, -4.0)
        return actor

    def make_critic(self, features_extractor=None) -> ContinuousCritic:
        extractor = PrivilegedCriticExtractor(
            self.observation_space, features_dim=256)
        kwargs = self._update_features_extractor(self.critic_kwargs, extractor)
        return ContinuousCritic(**kwargs).to(self.device)


class PriorReplayDataset:
    """Immutable replay-complete prior loaded from a validated manifest."""

    REQUIRED = (
        "obs_actor", "obs_privileged", "actions", "rewards",
        "next_actor", "next_privileged", "dones", "category", "bc_mask",
    )

    def __init__(self, manifest: str | Path, device: str | th.device = "cpu"):
        self.manifest_path = Path(manifest)
        payload = json.loads(self.manifest_path.read_text())
        shards = payload.get("shards", [])
        if not shards:
            raise ValueError("prior manifest contains no shards")
        arrays: dict[str, list[np.ndarray]] = {key: [] for key in self.REQUIRED}
        for item in shards:
            path = Path(item["path"])
            if not path.is_absolute():
                path = self.manifest_path.parent / path
            with np.load(path, allow_pickle=False) as shard:
                missing = [key for key in self.REQUIRED if key not in shard]
                if missing:
                    raise ValueError(f"prior shard {path} missing {missing}")
                n = int(shard["actions"].shape[0])
                if any(int(shard[key].shape[0]) != n for key in self.REQUIRED):
                    raise ValueError(f"prior shard {path} has inconsistent rows")
                for key in self.REQUIRED:
                    arrays[key].append(np.asarray(shard[key]))
        self.data = {key: np.concatenate(parts, axis=0) for key, parts in arrays.items()}
        self.device = th.device(device)
        self.size = int(self.data["actions"].shape[0])
        if self.size == 0:
            raise ValueError("prior replay is empty")
        category = self.data["category"].astype(np.int64).reshape(-1)
        # 0=failure boundary, 1=success, 2=recovery. Oversample recovery/success.
        self.weights = np.where(category == 2, 3.0, np.where(category == 1, 2.0, 1.0))
        self.weights = self.weights / self.weights.sum()
        self.rng = np.random.default_rng(int(payload.get("sampling_seed", 20260712)))

    def sample(self, n: int) -> tuple[DictReplayBufferSamples, th.Tensor]:
        idx = self.rng.choice(self.size, size=int(n), replace=True, p=self.weights)
        tensor = lambda key: th.as_tensor(self.data[key][idx], device=self.device)
        observations = {
            "actor": tensor("obs_actor").float(),
            "privileged": tensor("obs_privileged").float(),
        }
        next_observations = {
            "actor": tensor("next_actor").float(),
            "privileged": tensor("next_privileged").float(),
        }
        samples = DictReplayBufferSamples(
            observations=observations,
            actions=tensor("actions").float(),
            next_observations=next_observations,
            dones=tensor("dones").float().reshape(-1, 1),
            rewards=tensor("rewards").float().reshape(-1, 1),
            discounts=None,
        )
        return samples, tensor("bc_mask").bool().reshape(-1, 1)


def _concat_samples(
    online: DictReplayBufferSamples,
    prior: DictReplayBufferSamples,
) -> DictReplayBufferSamples:
    cat_dict = lambda a, b: {key: th.cat([a[key], b[key]], dim=0) for key in a}
    discounts = None
    if online.discounts is not None and prior.discounts is not None:
        discounts = th.cat([online.discounts, prior.discounts], dim=0)
    return DictReplayBufferSamples(
        observations=cat_dict(online.observations, prior.observations),
        actions=th.cat([online.actions, prior.actions], dim=0),
        next_observations=cat_dict(online.next_observations, prior.next_observations),
        dones=th.cat([online.dones, prior.dones], dim=0),
        rewards=th.cat([online.rewards, prior.rewards], dim=0),
        discounts=discounts,
    )


class RLPDSAC(SAC):
    """SAC with annealed immutable-prior/fresh replay and early BC auxiliary."""

    def __init__(
        self,
        *args,
        prior_manifest: str | Path,
        prior_ratio_start: float = 0.50,
        prior_ratio_end: float = 0.20,
        prior_anneal_steps: int = 300_000,
        bc_coef_start: float = 0.10,
        bc_decay_steps: int = 100_000,
        **kwargs,
    ):
        self._prior_manifest_arg = str(prior_manifest)
        self.prior_ratio_start = float(prior_ratio_start)
        self.prior_ratio_end = float(prior_ratio_end)
        self.prior_anneal_steps = int(prior_anneal_steps)
        self.bc_coef_start = float(bc_coef_start)
        self.bc_decay_steps = int(bc_decay_steps)
        super().__init__(*args, **kwargs)
        self.prior_replay = PriorReplayDataset(prior_manifest, device=self.device)

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + ["prior_replay"]

    def _prior_ratio(self) -> float:
        frac = min(1.0, self.num_timesteps / max(1, self.prior_anneal_steps))
        return self.prior_ratio_start + frac * (
            self.prior_ratio_end - self.prior_ratio_start)

    def _mixed_sample(self, batch_size: int):
        n_prior = int(np.clip(round(batch_size * self._prior_ratio()), 1, batch_size - 1))
        n_online = batch_size - n_prior
        online = self.replay_buffer.sample(  # type: ignore[union-attr]
            n_online, env=self._vec_normalize_env)
        prior, prior_bc = self.prior_replay.sample(n_prior)
        mixed = _concat_samples(online, prior)
        bc_mask = th.cat([
            th.zeros((n_online, 1), dtype=th.bool, device=self.device),
            prior_bc,
        ], dim=0)
        return mixed, bc_mask

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)
        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses, bc_losses = [], [], []

        for gradient_step in range(gradient_steps):
            replay_data, bc_mask = self._mixed_sample(batch_size)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma
            if self.use_sde:
                self.actor.reset_noise()
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)
            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(
                    self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(float(ent_coef_loss.item()))
            else:
                ent_coef = self.ent_coef_tensor
            ent_coefs.append(float(ent_coef.item()))
            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(
                    replay_data.next_observations)
                next_q = th.cat(self.critic_target(
                    replay_data.next_observations, next_actions), dim=1)
                next_q = th.min(next_q, dim=1, keepdim=True).values
                next_q = next_q - ent_coef * next_log_prob.reshape(-1, 1)
                target_q = replay_data.rewards + (
                    1 - replay_data.dones) * discounts * next_q
            current_q = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = 0.5 * sum(F.mse_loss(q, target_q) for q in current_q)
            critic_losses.append(float(critic_loss.item()))
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            q_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            actor_loss = (ent_coef * log_prob - th.min(q_pi, dim=1, keepdim=True).values).mean()
            bc_loss = th.zeros((), device=self.device)
            bc_frac = max(0.0, 1.0 - self.num_timesteps / max(1, self.bc_decay_steps))
            bc_coef = self.bc_coef_start * bc_frac
            if bc_coef > 0.0 and bool(bc_mask.any()):
                deterministic = self.actor(replay_data.observations, deterministic=True)
                mask = bc_mask.squeeze(1)
                bc_loss = F.mse_loss(deterministic[mask], replay_data.actions[mask])
                actor_loss = actor_loss + bc_coef * bc_loss
            actor_losses.append(float(actor_loss.item()))
            bc_losses.append(float(bc_loss.item()))
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/bc_loss", np.mean(bc_losses))
        self.logger.record("train/prior_ratio", self._prior_ratio())
        if ent_coef_losses:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))


__all__ = [
    "ActorHistoryExtractor", "PrivilegedCriticExtractor", "AsymmetricSACPolicy",
    "PriorReplayDataset", "RLPDSAC",
]
