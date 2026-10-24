import math

import numpy as np
import pytest
import torch

from lunar_rl.agents.ppo import ActorCritic, compute_gae


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
