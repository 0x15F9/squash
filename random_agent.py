import gym
import gym_squash

env = gym.make('Squash-v0')

state = env.reset()

print(state.shape)