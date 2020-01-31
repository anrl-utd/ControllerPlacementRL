from gym.envs.registration import register

register(
    id='Controller-v0',
    entry_point='controller_env.envs:ControllerEnv',
)

register(
	id='Controller-RandomPlacement-v0',
	entry_point='controller_env.envs:ControllerRandomStart',
)

register(
    id='Controller-Direct-v0',
    entry_piont='controller_env.envs:ControllerDirectSelect',
)