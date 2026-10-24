import numpy as np
import pytest
import torch
from torch import nn

from lunar_rl.agents import load_agent
from lunar_rl.agents.dqn import DQNAgent, DQNConfig, linear_epsilon, td_targets
from lunar_rl.agents.replay_buffer import Batch


class FixedQ(nn.Module):
    """Returns a fixed Q-table row for every input, to make targets hand-checkable."""

    def __init__(self, values):
        super().__init__()
        self.values = torch.tensor(values, dtype=torch.float32)

    def forward(self, obs):
        return self.values.expand(obs.shape[0], -1)


def test_td_target_terminal_vs_bootstrap():
    rewards = torch.tensor([1.0, 1.0])
    next_obs = torch.zeros(2, 8)
    # Second transition: the episode was *truncated* by the time limit, so terminated=0
    # and it must still bootstrap.
    terminated = torch.tensor([1.0, 0.0])
    q_target = FixedQ([0.0, 5.0, 2.0, 1.0])
    out = td_targets(rewards, next_obs, terminated, q_target, gamma=0.9)
    torch.testing.assert_close(out, torch.tensor([1.0, 1.0 + 0.9 * 5.0]))


def test_double_dqn_uses_online_argmax_and_target_value():
    rewards, next_obs, terminated = torch.zeros(1), torch.zeros(1, 8), torch.zeros(1)
    q_target = FixedQ([0.0, 5.0, 2.0, 1.0])  # vanilla would pick action 1 (value 5)
    q_online = FixedQ([0.0, 1.0, 3.0, 0.0])  # online net prefers action 2
    vanilla = td_targets(rewards, next_obs, terminated, q_target, gamma=1.0)
    double = td_targets(rewards, next_obs, terminated, q_target, gamma=1.0, q_online=q_online)
    assert vanilla.item() == 5.0
    assert double.item() == 2.0  # Q_target(s', argmax Q_online) = Q_target[2]


def test_linear_epsilon_schedule():
    assert linear_epsilon(0, 1.0, 0.1, 100) == 1.0
    assert linear_epsilon(50, 1.0, 0.1, 100) == pytest.approx(0.55)
    assert linear_epsilon(100, 1.0, 0.1, 100) == pytest.approx(0.1)
    assert linear_epsilon(10_000, 1.0, 0.1, 100) == pytest.approx(0.1)


def _random_batch(n=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return Batch(torch.randn(n, 8, generator=g), torch.randint(0, 4, (n,), generator=g),
                 torch.randn(n, generator=g), torch.randn(n, 8, generator=g),
                 torch.randint(0, 2, (n,), generator=g).float())


def test_target_network_only_changes_on_sync():
    torch.manual_seed(0)
    agent = DQNAgent(cfg=DQNConfig(hidden_sizes=[16]))
    before = [p.clone() for p in agent.q_target.parameters()]
    agent.update(_random_batch())
    # Online network moved, target network did not.
    assert any(not torch.equal(a, b) for a, b in zip(agent.q.parameters(), before))
    assert all(torch.equal(a, b) for a, b in zip(agent.q_target.parameters(), before))
    agent.sync_target()
    assert all(torch.equal(a, b) for a, b in zip(agent.q.parameters(), agent.q_target.parameters()))


def test_update_reduces_loss_on_fixed_batch():
    torch.manual_seed(0)
    agent = DQNAgent(cfg=DQNConfig(hidden_sizes=[32], learning_rate=1e-2))
    batch = _random_batch(64)
    first = agent.update(batch)["loss"]
    for _ in range(100):
        last = agent.update(batch)["loss"]
    assert last < first


def test_act_greedy_vs_epsilon():
    agent = DQNAgent(cfg=DQNConfig(hidden_sizes=[16]))
    obs = np.zeros(8, dtype=np.float32)
    greedy = {agent.act(obs, deterministic=True) for _ in range(20)}
    assert len(greedy) == 1  # deterministic
    agent.epsilon = 1.0
    explore = {agent.act(obs, deterministic=False) for _ in range(200)}
    assert explore == {0, 1, 2, 3}  # epsilon=1 -> uniform random


def test_save_load_round_trip(tmp_path):
    agent = DQNAgent(cfg=DQNConfig(hidden_sizes=[16], double_dqn=True))
    path = tmp_path / "dqn.pt"
    agent.save(path)
    loaded = load_agent("dqn", path)
    assert loaded.cfg == agent.cfg
    obs = np.random.default_rng(0).normal(size=(20, 8)).astype(np.float32)
    assert [agent.act(o) for o in obs] == [loaded.act(o) for o in obs]


def test_without_target_network_targets_track_the_online_net():
    torch.manual_seed(0)
    agent = DQNAgent(cfg=DQNConfig(hidden_sizes=[16], use_target_network=False, gamma=0.9))
    batch = _random_batch()
    agent.update(batch)  # online net moves, target net stays at init
    expected = td_targets(batch.rewards, batch.next_obs, batch.terminated, agent.q, 0.9)
    stale = td_targets(batch.rewards, batch.next_obs, batch.terminated, agent.q_target, 0.9)
    assert not torch.allclose(expected, stale)
    # The next update must regress towards the online-net targets: check via the loss it reports.
    q_sa = agent.q(batch.obs).gather(1, batch.actions.unsqueeze(1)).squeeze(1)
    loss_online = torch.nn.functional.smooth_l1_loss(q_sa, expected).item()
    assert agent.update(batch)["loss"] == pytest.approx(loss_online, rel=1e-5)
