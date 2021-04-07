import gym
import gym_squash

env = gym.make('BreakoutNoFrameskip-v0')

env.reset()

for i in range(100):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    env.render(mode='human')
    print(i)
    if done:
        env.reset()
    