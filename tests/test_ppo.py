import math

import numpy as np
import pytest
import torch

from lunar_rl.agents import load_agent
from lunar_rl.agents.ppo import (
    ActorCritic,
    PPOAgent,
    PPOConfig,
    Rollout,
    clipped_policy_loss,
    compute_gae,
    explained_variance,
)


def test_gae_matches_hand_computation():
    # One env, 2 steps, no episode end: hand-computed with gamma=0.5, lambda=0.5.
    rewards = np.array([[1.0], [2.0]])
    values = np.array([[0.5], [1.0]])
    dones = np.zeros((2, 1))
    last_values = np.array([4.0])
    adv, ret = compute_gae(rewards, values, dones, last_values, gamma=0.5, lam=0.5)
    delta1 = 2.0 + 0.5 * 4.0 - 1.0          # 3.0
    delta0 = 1.0 + 0.5 * 1.0 - 0.5          # 1.0
    np.testing.assert_allclose(adv[:, 0], [delta0 + 0.25 * delta1, delta1])
    np.testing.assert_allclose(ret, adv + values)


def test_gae_stops_at_episode_end():
    rewards = np.array([[1.0], [10.0]])
    values = np.array([[0.0], [5.0]])
    dones = np.array([[1.0], [0.0]])         # episode ends at t=0; t=1 is a new episode
    adv, _ = compute_gae(rewards, values, dones, np.array([0.0]), gamma=0.9, lam=0.9)
    assert adv[0, 0] == pytest.approx(1.0)  # no bootstrap from the next episode's value


def test_gae_lambda_one_is_discounted_return_minus_value():
    rng = np.random.default_rng(0)
    rewards = rng.normal(size=(6, 1))
    values = rng.normal(size=(6, 1))
    dones = np.zeros((6, 1))
    last_values = np.array([0.3])
    adv, ret = compute_gae(rewards, values, dones, last_values, gamma=0.9, lam=1.0)
    mc = [sum(0.9 ** k * rewards[t + k, 0] for k in range(6 - t)) + 0.9 ** (6 - t) * 0.3 for t in range(6)]
    np.testing.assert_allclose(ret[:, 0], mc, rtol=1e-5)


def test_initial_policy_is_near_uniform():
    torch.manual_seed(0)
    ac = ActorCritic(8, 4, [64, 64])
    entropy = ac.dist(torch.randn(256, 8)).entropy()
    assert entropy.min().item() > 0.99 * math.log(4)  # small (0.01) policy-head gain


def test_evaluate_actions_shapes():
    ac = ActorCritic(8, 4, [16])
    logp, ent, v = ac.evaluate_actions(torch.randn(5, 8), torch.tensor([0, 1, 2, 3, 0]))
    assert logp.shape == ent.shape == v.shape == (5,)


def test_clipped_loss_caps_the_ratio():
    old = torch.zeros(4)
    # ratios: 1.5 and 0.5 (outside [0.8, 1.2]), 1.1 and 0.9 (inside)
    new = torch.log(torch.tensor([1.5, 0.5, 1.1, 0.9]))
    adv = torch.tensor([1.0, -1.0, 1.0, -1.0])
    loss, stats = clipped_policy_loss(new, old, adv, clip_range=0.2)
    # Positive advantage, ratio 1.5 -> capped at 1.2. Negative advantage, ratio 0.5 -> capped at 0.8.
    expected = -(1.2 * 1.0) + (0.8 * 1.0) - (1.1 * 1.0) + (0.9 * 1.0)
    assert loss.item() == pytest.approx(expected / 4)
    assert stats["clip_fraction"] == pytest.approx(0.5)
    assert stats["approx_kl"] > 0


def test_clipped_loss_keeps_pessimistic_unclipped_term():
    # Positive advantage but ratio *below* 1 - clip: the unclipped term is the larger loss and is kept,
    # so the gradient still pushes the ratio back up (clipping only removes the incentive to over-move).
    new = torch.log(torch.tensor([0.5])).requires_grad_()
    loss, _ = clipped_policy_loss(new, torch.zeros(1), torch.tensor([1.0]), clip_range=0.2)
    assert loss.item() == pytest.approx(-0.5)
    loss.backward()
    assert new.grad.item() != 0


def test_explained_variance():
    r = np.array([1.0, 2.0, 3.0])
    assert explained_variance(r, r) == pytest.approx(1.0)
    assert explained_variance(np.full(3, r.mean()), r) == pytest.approx(0.0)


def _rollout(n=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    agent = PPOAgent(cfg=PPOConfig(hidden_sizes=[16], batch_size=64))
    obs = torch.randn(n, 8, generator=g)
    with torch.no_grad():
        d = agent.policy.dist(obs)
        actions = d.sample()
        values = agent.policy.value(obs)
    returns = torch.randn(n, generator=g)
    return agent, Rollout(obs, actions, d.log_prob(actions), returns - values, returns, values)


def test_update_returns_diagnostics():
    agent, rollout = _rollout()
    stats = agent.update(rollout)
    for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "explained_variance"):
        assert k in stats
    assert stats["approx_kl"] >= 0


def test_act_deterministic_is_argmax_and_save_load(tmp_path):
    agent = PPOAgent(cfg=PPOConfig(hidden_sizes=[16]))
    obs = np.random.default_rng(0).normal(size=(20, 8)).astype(np.float32)
    with torch.no_grad():
        argmax = agent.policy.actor(torch.as_tensor(obs)).argmax(1).tolist()
    assert [agent.act(o) for o in obs] == argmax
    agent.save(tmp_path / "ppo.pt")
    loaded = load_agent("ppo", tmp_path / "ppo.pt")
    assert loaded.cfg == agent.cfg
    assert [loaded.act(o) for o in obs] == argmax
