from gym.envs.registration import register

register(
    id='Squash-v0',
    entry_point='gym_squash.envs:SquashEnv',
)