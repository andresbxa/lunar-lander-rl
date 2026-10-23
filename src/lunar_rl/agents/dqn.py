"""DQN agent implemented from scratch in PyTorch.

TODO (phase 2):
    - Q-network (MLP) and target network with periodic hard / soft (Polyak) updates
    - Replay buffer with uniform sampling
    - Epsilon-greedy exploration with linear decay
    - Huber loss, gradient clipping
    - Optional: Double DQN, Dueling heads (as ablations in phase 4)
    - Implement the `Agent` protocol (`act(obs, deterministic)`) so it plugs into
      `lunar_rl.evaluate` and `lunar_rl.record` unchanged.
"""
