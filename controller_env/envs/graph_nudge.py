import numpy as np
from gym import spaces
from controller_env.envs.graph_env import ControllerEnv
class ControllerRandomStart(ControllerEnv):
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
		self.action_space = spaces.Box(np.zeros(len(clusters)), np.ones(len(clusters)), dtype=np.uint8)
		self.observation_space = spaces.Box(np.zeros(shape=len(graph.nodes)), np.ones(shape=len(graph.nodes)), dtype=np.bool)
		self.controllers = [np.random.choice(i) for i in self.clusters]

	def step(self, action):
		"""
		Steps environment once.
		Args:
			action: Action to perform
					List of size controllers
					[[0, 0.5, 1, 0.5, 1], [1, 0, 0, 0, 0], ...] example for graph with degree 5
		Returns:
			Tuple of (State, Reward) after selecting controllers and passing to base environment (latency for 1000 packets)
			State is a boolean array of size <number of switches> which indicates whether a switch is a controller or not
		"""
		for controller_index in range(len(action)):
			if(action[controller_index] is 1):
				neighbors = self.graph.neighbors(self.controllers[controller_index])
				neigh = list(neighbors)
				choice = np.random.choice(neigh)
				while(controller_index != self.graph.nodes[choice]['cluster']):
					choice = np.random.choice(neigh)
				self.controllers[controller_index] = choice
		#Construct the state (boolean array of size <number of switches> indicating whether a switch is a controller)
		state = np.zeros(shape=len(self.graph.nodes))
		state[self.controllers] = 1
		return (state, super().step(self.controllers))