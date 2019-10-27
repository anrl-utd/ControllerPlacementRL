from gym.envs.registration import register

register(
    id='Controller-v0',
    entry_point='controller_env.envs:ControllerEnv',
)