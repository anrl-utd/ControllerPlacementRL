import numpy as np
from gym import spaces
import itertools
from controller_env.envs.graph_env import ControllerEnv
class ControllerSlowSelect(ControllerEnv):
	"""
	Environment that selects controllers one at a time, ends episode when all are selected
	Input: Graph with clusters
	Action: List of size controllers with elements that are a list of size largest degree of graph and probabilities for each neighbor to be controller.
			Argmax across lists to choose index of neighbor to nudge controller to
	State: List of whether each node is controller or not [0, 1, ..., 0]
	Reward: ControllerEnv reward (sum of latency for simulating 1000 packets)
	"""
	def __init__(self, graph, clusters, pos=None):
		"""Initilizes environment and assigns random nodes to be controllers"""
		super().__init__(graph, clusters, pos, check_controller_num=False)
		self.action_space = spaces.Discrete(len(graph.nodes))
		#self.action_space = spaces.Box(0, 1, (len(graph.nodes),), dtype=np.float32)
		self.observation_space = spaces.Box(0, 1, (len(graph.nodes),), dtype=np.bool)
		self.controllers = []
		self.num_clusters = clusters.shape[0]

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
		self.controllers.append(action)
		#Construct the state (boolean array of size <number of switches> indicating whether a switch is a controller)
		state = np.zeros(shape=len(self.graph.nodes))
		state[self.controllers] = 1
		(obs, rew, done, i) = (state, super().step(self.controllers), len(self.controllers) >= self.num_clusters, {})
		return (obs, -rew, done, i)

	def reset(self):
		self.controllers = []
		state = np.zeros(shape=len(self.graph.nodes))
		return state