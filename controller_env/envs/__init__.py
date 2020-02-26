from controller_env.envs.graph_env import ControllerEnv
from controller_env.envs.graph_nudge import ControllerRandomStart
from controller_env.envs.graph_direct import ControllerDirectSelect
from controller_env.envs.graph_remove import ControllerRemove
from gym.envs.registration import register
import gym

register(
    id='ControllerRemoveNodes-v0',
    entry_point='controller_env.envs:ControllerRemove'
)