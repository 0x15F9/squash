import gym
import gym_squash

env = gym.make('Squash-v0')

print('action space:', env.action_space.n)
print('observation space:', env.observation_space.shape)
print(env.get_action_meanings())

# state = env.reset()

# print(state.shape)