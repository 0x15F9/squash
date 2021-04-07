import gym
from gym import error, spaces, utils
from gym.utils import seeding

rendering = None
def checkRendering():
  global rendering
  if rendering is None:
    from gym.envs.classic_control import rendering as rendering

import numpy as np
import cv2
from time import sleep
import random

config = {
  'screen_width': 160,
  'screen_height': 210,
  'paddle_width': 15,
  'paddle_height': 5,
  'paddle_y': 190,
  'ball_width': 2,
  'ball_height': 2,
  'score_max': 20,
  'top_wall': 45,
  'wall_w': 8,
}

class Rect:
  def __init__(self, x, y, w, h):
    self.x = x
    self.y = y
    self.w = w
    self.h = h
  
  def str(self):
    return (self.x, self.y)
    
class Window:
  def __init__(self, w, h):
    self.w = w
    self.h = h
    
class TopWall(Rect):
  def __init__(self):
    super().__init__(0, config['top_wall'], config['screen_width'], config['paddle_height'])
    
class SideWall(Rect):
  def __init__(self, x):
    super().__init__(x, config['top_wall'], config['wall_w'], config['screen_height']-config['top_wall'])
    
class Paddle(Rect):
  def __init__(self, x, y, w, h, v=3):
    super().__init__(x, y, w, h)
    self.v = v
    
  def step(self, action):
    if action == 'right':
      self.x += self.v
    elif action == 'left':
      self.x -= self.v

class Ball(Rect):
  def __init__(self, x, y, w, h, v=1, a=0, down=1, right=1):
    super().__init__(x, y, w, h)
    self.v = v
    self.a = random.randint(-v, +v)
    self.down = down
    self.right = right
    
  def mirror_v(self):
    self.down = self.down * -1
    
  def mirror_h(self):
    self.right = self.right * -1
    
  def randomize_a(self):
    self.a = random.randint(-self.v, self.v)
    
  def step(self):
    self.y += self.v * self.down
    self.x += self.a * self.right

class SquashEnv(gym.Env):
  metadata = {
    'render.modes': ['human', 'rgb_array', 'state'],
    'video.frames_per_second' : 50
  }

  # for compatibility with typical atari wrappers
  atari_action_meaning = {
    0: "NOOP",
    1: "FIRE",
    2: "UP",
    3: "RIGHT",
    4: "LEFT",
    5: "DOWN",
    6: "UPRIGHT",
    7: "UPLEFT",
    8: "DOWNRIGHT",
    9: "DOWNLEFT",
    10: "UPFIRE",
    11: "RIGHTFIRE",
    12: "LEFTFIRE",
    13: "DOWNFIRE",
    14: "UPRIGHTFIRE",
    15: "UPLEFTFIRE",
    16: "DOWNRIGHTFIRE",
    17: "DOWNLEFTFIRE",
  }
  atari_action_set = {
    0, # NOOP
    1, # FIRE
    3, # RIGHT
    4, # LEFT
  }
  
  SCREEN_W = config['screen_width']
  SCREEN_H = config['screen_height']
  PADDLE_W = config['paddle_width']
  PADDLE_H = config['paddle_height']
  PADDLE_Y = config['paddle_y']
  BALL_W = config['ball_width']
  BALL_H = config['ball_height']
  WALL_W = config['wall_w']
  SCORE_MAX = config['score_max']

  def __init__(self):
    self.action_space = spaces.Discrete(len(self.atari_action_set))
    self.observation_space = spaces.Box(low=0, high=255, 
                shape=(self.SCREEN_W, self.SCREEN_H, 3), dtype=np.uint8)
    
    self.reset()

  
  def step(self, action):
    reward = 0
    
    # Paddle action
    if action == 2 and self.paddle.x+self.paddle.w < self.SCREEN_W:
      self.paddle.step('right')
    elif action == 3 and self.paddle.x > 0:
      self.paddle.step('left')
      
    # Ball action
    self.ball.step()
    # Check collision with side walls
    if self.ball.x <= self.left_wall.x+self.WALL_W or self.ball.x+self.BALL_W >= self.right_wall.x:  
      self.ball.mirror_h()
    # Check collision with top wall
    elif self.ball.y <= self.top_wall.y+self.top_wall.h:                                
      self.ball.y = self.top_wall.y+self.top_wall.h
      self.ball.mirror_v()
      self.ball.randomize_a()      
      reward = 1
    elif self.ball.y+self.BALL_H >= self.paddle.y:
      # Check collision with Paddle
      if self.ball.x < self.paddle.x + self.PADDLE_W and self.ball.x + self.BALL_W > self.paddle.x:
        self.ball.mirror_v()
        self.ball.randomize_a()      
      # Check Ball go past Paddle
      else:
        reward = -1
      
    self.score += reward
    done = True if self.score == self.SCORE_MAX or reward == -1 else False
    info = self.get_info()
    
    return self.get_obs(), reward, done, info


  def reset(self):
    self.score = 0
    self.viewer = None
    self.window = Window(self.SCREEN_W, self.SCREEN_H)
    self.top_wall = TopWall()
    self.left_wall = SideWall(0)
    self.right_wall = SideWall(self.SCREEN_W-self.WALL_W)
    self.paddle = Paddle((self.SCREEN_W-self.PADDLE_W)/2, self.PADDLE_Y, self.PADDLE_W, self.PADDLE_H)
    self.ball = Ball((self.SCREEN_W+self.BALL_W)/2, (self.SCREEN_H+self.BALL_H)/2, self.BALL_W, self.BALL_H)
    return self.get_obs()

  def get_info(self):
    return {
        'score': self.score,
        'paddle': self.paddle.str(),
        'ball': self.ball.str()
      }
    
  def get_obs(self):
    # 0 is black, 255 is white
    canvas = np.zeros((self.SCREEN_W, self.SCREEN_H, 3), dtype=np.uint8)
    # draw wall
    canvas[
      int(self.top_wall.x):int(self.top_wall.x+self.top_wall.w+1),
      int(self.top_wall.y):int(self.top_wall.y+self.top_wall.h+1),
      ] = 255
    canvas[
      int(self.left_wall.x):int(self.left_wall.x+self.left_wall.w+1),
      int(self.left_wall.y):int(self.left_wall.y+self.left_wall.h+1),
      ] = 255
    canvas[
      int(self.right_wall.x):int(self.right_wall.x+self.right_wall.w+1),
      int(self.right_wall.y):int(self.right_wall.y+self.right_wall.h+1),
      ] = 255
    # draw paddle
    canvas[
      int(self.paddle.x):int(self.paddle.x+self.PADDLE_W+1),
      int(self.paddle.y):int(self.paddle.y+self.PADDLE_H+1),
      ] = 255
    # draw ball
    canvas[
      int(self.ball.x):int(self.ball.x+self.BALL_W+1),
      int(self.ball.y):int(self.ball.y+self.BALL_H+1),
      ] = 255
    return np.swapaxes(canvas, 0, 1)

  def render(self, mode='rgb_array', delay=0.03, close=False):
    if mode == 'state':
      return self.get_info()
    elif mode == 'rgb_array' or mode == 'human':
      if self.viewer is None:
        checkRendering()
        self.viewer = rendering.SimpleImageViewer()
      self.viewer.imshow(self.get_obs())
      if mode == 'human':
        sleep(delay)
      return True
    else:
      super(SquashEnv, self).render(mode=mode) # just raise an exception


  def close(self):
    if self.viewer is not None:
      self.viewer.close()
    
  
  def get_action_meanings(self):
    return [self.atari_action_meaning[i] for i in self.atari_action_set]
