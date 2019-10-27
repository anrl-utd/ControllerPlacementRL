import gym
from gym import error, spaces, utils
from gym.utils import seeding

class ControllerEnv(gym.Env):
	metadata = {'render.modes' : ['human']}

	def __init__(self):
		print("Initialized environment!")

	def step(self, action):
		print("Environment step")
		
	def reset(self):
		print("Reset environment")

	def render(self, mode='human', close=False):
		print("Rendered enviroment")