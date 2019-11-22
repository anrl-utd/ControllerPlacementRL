import numpy as np
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
		self.controllers = [np.random.choice(i) for i in self.clusters]

	def step(self, action):
		"""
		Steps environment once.
		Args:
			action: Action to perform
					List of size controllers
					[[0, 0.5, 1, 0.5, 1], [1, 0, 0, 0, 0], ...] example for graph with degree 5
		Returns:
			Reward after selecting controllers and passing to base environment (latency for 1000 packets)
		"""
		print(action)
		for controller_index in range(len(action)):
			if(action[controller_index] is 1):
				neighbors = self.graph.neighbors(self.controllers[controller_index])
				self.controllers[controller_index] = np.random.choice(list(neighbors))
		return super().step(self.controllers)