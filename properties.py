import gym
import gym_squash

env = gym.make('Squash-v0')
import numpy as np


print('action space:', env.action_space.n)
print('observation space:', env.observation_space.shape)
print(env.get_action_meanings())

for _ in range(100):
# while True:
    # action = env.action_space.sample()
    action = 0
    observation, reward, done, info = env.step(action)
    # print(info)
    env.render(mode='human')
    if done:
        break
   
# print(np.sum(observation))
# cv2.imshow("frame", observation) 
# cv2.waitKey(0)

# action = env.action_space.sample()
# observation, reward, done, info = env.step(action)

# state = env.reset()

# print(state.shape)