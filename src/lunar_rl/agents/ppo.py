"""PPO agent implemented from scratch in PyTorch.

TODO (phase 3):
    - Actor-critic MLP (separate or shared trunk)
    - Rollout buffer over vectorized envs, GAE(lambda) advantages
    - Clipped surrogate objective, value loss, entropy bonus
    - Advantage normalization, gradient clipping, learning-rate annealing
    - Implement the `Agent` protocol (`act(obs, deterministic)`) and compare
      against the SB3 baseline using the same configs / evaluation seeds.
"""
