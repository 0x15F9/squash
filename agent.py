import gym
import gym_squash

import random

env = gym.make('Squash-v0')

env.reset()
action = 0 #env.action_space.sample()

# for i in range(500):
#     for j in range(-3,3):
#         obs, reward, done, info = env.step(action)
#         env.render(mode='human', delay=.005)
#         paddle_x, paddle_y = info['paddle']
#         ball_x, ball_y = info['ball']
#         action = 3 if paddle_x+3*j>ball_x else 2
#         if done:
#             env.reset()
    
action = 0
for i in range(5000):
    obs, reward, done, info = env.step(action)
    env.render(mode='human', delay=.03)
    paddle_x, paddle_y = info['paddle']
    ball_x, ball_y = info['ball']
    action = 3 if paddle_x+14>ball_x else 2
    if done:
        print()
        env.reset()