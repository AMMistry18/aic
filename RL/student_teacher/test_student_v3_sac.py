import gymnasium as gym
import numpy as np
import torch
import json

from RL.student_teacher.student_v3_sac import AsymmetricSACPolicy, PriorReplayDataset


def _policy():
    obs_space = gym.spaces.Dict({
        "actor": gym.spaces.Box(-np.inf, np.inf, (8, 48), np.float32),
        "privileged": gym.spaces.Box(-np.inf, np.inf, (32,), np.float32),
    })
    action_space = gym.spaces.Box(-1.0, 1.0, (6,), np.float32)
    return AsymmetricSACPolicy(
        obs_space, action_space, lambda _: 3e-4,
        net_arch=[128, 128], share_features_extractor=False)


def test_actor_is_zero_initialized_and_privileged_invariant():
    policy = _policy()
    actor = torch.randn(4, 8, 48)
    obs_a = {"actor": actor, "privileged": torch.zeros(4, 32)}
    obs_b = {"actor": actor, "privileged": torch.randn(4, 32) * 100.0}
    with torch.no_grad():
        action_a = policy.actor(obs_a, deterministic=True)
        action_b = policy.actor(obs_b, deterministic=True)
    assert torch.equal(action_a, torch.zeros_like(action_a))
    assert torch.equal(action_a, action_b)


def test_critic_uses_privileged_state():
    policy = _policy()
    actor = torch.randn(4, 8, 48)
    actions = torch.zeros(4, 6)
    obs_a = {"actor": actor, "privileged": torch.zeros(4, 32)}
    obs_b = {"actor": actor, "privileged": torch.ones(4, 32)}
    with torch.no_grad():
        q_a = policy.critic(obs_a, actions)[0]
        q_b = policy.critic(obs_b, actions)[0]
    assert not torch.allclose(q_a, q_b)


def test_prior_manifest_is_replay_complete(tmp_path):
    n = 7
    shard = tmp_path / "prior.npz"
    np.savez_compressed(
        shard,
        obs_actor=np.zeros((n, 8, 48), np.float32),
        obs_privileged=np.zeros((n, 32), np.float32),
        actions=np.zeros((n, 6), np.float32),
        rewards=np.zeros(n, np.float32),
        next_actor=np.ones((n, 8, 48), np.float32),
        next_privileged=np.ones((n, 32), np.float32),
        dones=np.zeros(n, np.float32),
        category=np.array([0, 1, 2, 1, 2, 0, 1], np.int8),
        bc_mask=np.array([0, 1, 1, 1, 1, 0, 1], np.int8),
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"shards": [{"path": shard.name}]}))
    prior = PriorReplayDataset(manifest)
    sample, bc = prior.sample(5)
    assert sample.observations["actor"].shape == (5, 8, 48)
    assert sample.next_observations["privileged"].shape == (5, 32)
    assert bc.shape == (5, 1)
