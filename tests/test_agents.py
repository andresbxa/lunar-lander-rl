import numpy as np

from lunar_rl.agents import Agent, RandomAgent, load_agent


def test_random_agent_follows_protocol_and_is_seeded():
    obs = np.zeros(8, dtype=np.float32)
    a, b = RandomAgent(seed=3), RandomAgent(seed=3)
    assert isinstance(a, Agent)
    actions_a = [a.act(obs) for _ in range(20)]
    actions_b = [b.act(obs) for _ in range(20)]
    assert actions_a == actions_b
    assert set(actions_a) <= {0, 1, 2, 3}


def test_load_agent_random():
    assert isinstance(load_agent("random"), RandomAgent)
