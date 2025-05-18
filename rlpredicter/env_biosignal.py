import gym
from gym import spaces
import numpy as np

class BioSignalEnv(gym.Env):
    """
    EEG-based bio-signal prediction environment for reinforcement learning.
    The agent's goal is to minimize prediction error (e.g., MAE) on biosignal targets.
    """
    def __init__(self, X, y):
        super().__init__()
        self.X = X
        self.y = y
        self.index = 0

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(X.shape[1],), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.min(y, axis=0),
            high=np.max(y, axis=0),
            dtype=np.float32
        )

    def reset(self):
        self.index = 0
        return self.X[self.index]

    def step(self, action):
        true = self.y[self.index]
        error = np.abs(action - true)
        reward = -np.sum(error ** 2)  # Use negative MSE as reward
        self.index += 1
        done = self.index >= len(self.X)
        info = {"true": true, "pred": action}
        return self.X[self.index % len(self.X)], reward, done, info