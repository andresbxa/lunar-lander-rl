import numpy as np

from lunar_rl.envs import env_condition_name, make_env


def rollout(env, seed: int, actions: list[int]) -> np.ndarray:
    obs, _ = env.reset(seed=seed)
    trajectory = [obs]
    for a in actions:
        obs, _, terminated, truncated, _ = env.step(a)
        trajectory.append(obs)
        if terminated or truncated:
            break
    return np.stack(trajectory)


def test_spaces():
    env = make_env(seed=0)
    assert env.observation_space.shape == (8,)
    assert env.action_space.n == 4
    assert env.spec.max_episode_steps == 1000
    env.close()


def test_same_seed_is_reproducible():
    actions = [0, 1, 2, 3] * 10
    np.testing.assert_array_equal(
        rollout(make_env(), seed=123, actions=actions),
        rollout(make_env(), seed=123, actions=actions),
    )


def test_different_seeds_differ():
    obs_a, _ = make_env().reset(seed=1)
    obs_b, _ = make_env().reset(seed=2)
    assert not np.allclose(obs_a, obs_b)


def test_seeded_action_space_is_reproducible():
    a = [make_env(seed=7).action_space.sample() for _ in range(5)]
    b = [make_env(seed=7).action_space.sample() for _ in range(5)]
    assert a == b


def test_wind_parameters_are_applied_and_reproducible():
    kwargs = dict(enable_wind=True, wind_power=10.0, turbulence_power=0.5)
    env = make_env(seed=0, **kwargs)
    base = env.unwrapped
    assert base.enable_wind is True
    assert base.wind_power == 10.0
    assert base.turbulence_power == 0.5

    actions = [0] * 50
    np.testing.assert_array_equal(rollout(env, 5, actions), rollout(make_env(**kwargs), 5, actions))


def test_wind_changes_dynamics():
    actions = [0] * 50
    calm = rollout(make_env(), 5, actions)
    windy = rollout(make_env(enable_wind=True, wind_power=20.0), 5, actions)
    assert not np.allclose(calm, windy)


def test_condition_name():
    assert env_condition_name(None) == "no_wind"
    assert env_condition_name({"enable_wind": False, "wind_power": 5}) == "no_wind"
    assert env_condition_name({"enable_wind": True, "wind_power": 10.0, "turbulence_power": 1.5}) == "wind10_turb1.5"
