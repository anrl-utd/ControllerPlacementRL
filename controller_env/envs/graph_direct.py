import numpy as np
from gym import spaces
import itertools
from controller_env.envs.graph_env import ControllerEnv
class ControllerDirectSelect(ControllerEnv):
	"""
	Environment that does 'nudging' of controllers (starts with random controller placement).
	Input: Graph with clusters
	Action: List of size controllers with elements that are a list of size largest degree of graph and probabilities for each neighbor to be controller.
			Argmax across lists to choose index of neighbor to nudge controller to
	State: List of whether each node is controller or not [0, 1, ..., 0]
	Reward: ControllerEnv reward (sum of latency for simulating 1000 packets)
	"""
	def __init__(self, graph, clusters, pos=None):
		"""Initilizes environment and assigns random nodes to be controllers"""
		super().__init__(graph, clusters, pos)
		#self.action_space = spaces.Box(0, 1, (len(clusters), self.degree), dtype=np.float32)
		self.action_space = spaces.Box(0, 1, (len(graph.nodes),), dtype=np.float32)
		self.observation_space = spaces.Box(0, 1, (len(graph.nodes),), dtype=np.bool)
		self.controllers = [np.random.choice(i) for i in self.clusters]
		self.original_controllers = self.controllers.copy()
		self.step_counter = 0

	def step(self, action):
		"""
		Steps environment once.
		Args:
			action: Action to perform
					List of size number of graph nodes
                    Similar to state, argmax of top (# of clusters) are labeled as controllers
					[0, 1, 0.9, 0.2, 0.3, ...] example for graph with degree 5
		Returns:
			Tuple of (State, Reward) after selecting controllers and passing to base environment (latency for 1000 packets)
			State is a boolean array of size <number of switches> which indicates whether a switch is a controller or not
		"""
		self.controllers = action.argsort()[-len(controllers)][::-1]
		#Construct the state (boolean array of size <number of switches> indicating whether a switch is a controller)
		state = np.zeros(shape=len(self.graph.nodes))
		state[self.controllers] = 1

		#Have episodes terminate after 100 steps
		done = False
		if(self.step_counter > 100):
			self.step_counter = 0
			done = True
		self.step_counter += 1
		return (state, super().step(self.controllers), done, {})

	def reset(self):
		self.controllers = self.original_controllers.copy()
		state = np.zeros(shape=len(self.graph.nodes))
		state[self.controllers] = 1
		return state