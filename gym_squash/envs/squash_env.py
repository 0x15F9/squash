import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np

class SquashEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    ...

  def step(self, action):
    ...

  def reset(self):
    return np.zeros(5)

  def render(self, mode='human'):
    ...

  def close(self):
    ...
