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
    entry_point='controller_env.envs:ControllerDirectSelect',
)

register(
    id='Controller-Select-v0',
    entry_point='controller_env.envs:ControllerSlowSelect',
)

register(
	id='Controller-Cluster-v0',
	entry_point='controller_env.envs:ControllerClusterSelect',
)

register(
	id='Controller-Cluster-Options-v0',
	entry_point='controller_env.envs:ControllerClusterSelectModified'
)

register(
	id='Controller-Single-v0',
	entry_point='controller_env.envs:ControllerSingleSelect'
)

register(
	id='Controller-All-v0',
	entry_point='controller_env.envs:ControllerAllSelect'
)