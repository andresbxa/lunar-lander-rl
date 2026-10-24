import numpy as np
import pytest
import torch

from lunar_rl.agents.replay_buffer import ReplayBuffer


def fill(buf: ReplayBuffer, n: int, start: int = 0) -> None:
    for k in range(start, start + n):
        obs = np.full(8, k, dtype=np.float32)
        buf.add(obs, action=k % 4, reward=float(k), next_obs=obs + 1, terminated=(k % 2 == 0))


def test_sample_shapes_and_dtypes():
    buf = ReplayBuffer(capacity=100, obs_dim=8, seed=0)
    fill(buf, 10)
    b = buf.sample(32)
    assert b.obs.shape == (32, 8) and b.obs.dtype == torch.float32
    assert b.next_obs.shape == (32, 8)
    assert b.actions.shape == (32,) and b.actions.dtype == torch.int64
    assert b.rewards.shape == (32,) and b.terminated.shape == (32,)


def test_transitions_stay_consistent():
    buf = ReplayBuffer(capacity=100, obs_dim=8, seed=0)
    fill(buf, 50)
    b = buf.sample(64)
    k = b.obs[:, 0]
    torch.testing.assert_close(b.rewards, k)
    torch.testing.assert_close(b.next_obs[:, 0], k + 1)
    torch.testing.assert_close(b.actions, k.long() % 4)
    torch.testing.assert_close(b.terminated, (k.long() % 2 == 0).float())


def test_only_filled_slots_are_sampled():
    buf = ReplayBuffer(capacity=100, obs_dim=8, seed=0)
    fill(buf, 5, start=1)  # values 1..5; untouched slots are all zeros
    assert len(buf) == 5
    assert (buf.sample(1000).obs[:, 0] >= 1).all()


def test_ring_buffer_overwrites_oldest():
    buf = ReplayBuffer(capacity=10, obs_dim=8, seed=0)
    fill(buf, 25)
    assert len(buf) == 10
    assert set(buf.sample(1000).obs[:, 0].long().tolist()) == set(range(15, 25))


def test_seeded_sampling_is_reproducible():
    a, b = ReplayBuffer(100, 8, seed=3), ReplayBuffer(100, 8, seed=3)
    fill(a, 50)
    fill(b, 50)
    torch.testing.assert_close(a.sample(16).obs, b.sample(16).obs)


def test_empty_buffer_raises():
    with pytest.raises(ValueError):
        ReplayBuffer(10, 8).sample(1)
