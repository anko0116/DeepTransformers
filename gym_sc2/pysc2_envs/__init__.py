from gym.envs.registration import register

register(
    id='defeat-zerglings-banelings-v0',
    entry_point='pysc2_envs.envs:DZBEnv',
)