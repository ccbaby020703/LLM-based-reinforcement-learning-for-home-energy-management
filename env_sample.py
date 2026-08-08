import gym
import numpy as np
from gym import spaces


class GYMEnv(gym.Env):
    def __init__(self):
        super(GYMEnv, self).__init__()
        self.state = 0
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-10, high=10, shape=(1,), dtype=np.float32)

    def step(self, action):
        self.state -= action[0][0]
        reward = -self.state ** 2
        done = False
        return [[self.state, 0, 0, 0]], [reward], [done], {}

    def reset(self):
        self.state = np.random.uniform(-10, 10)
        return [[self.state, 0, 0, 0]]

    def render(self, mode='human'):
        print(f"Current state: {self.state}")
