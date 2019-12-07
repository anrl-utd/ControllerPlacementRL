import numpy as np
from gym import spaces
import itertools
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
		#self.action_space = spaces.Box(0, 1, (len(clusters), self.degree), dtype=np.float32)
		self.action_space = spaces.Box(0, 1, (len(clusters) * self.degree,), dtype=np.float32)
		self.observation_space = spaces.Box(0, 1, (len(graph.nodes),), dtype=np.bool)
		self.controllers = [np.random.choice(i) for i in self.clusters]
		self.original_controllers = self.controllers.copy()

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
		act = np.split(action, 3)
		new_controller_neighbor = np.argmax(act, axis=1)
		for controller_index in range(len(act)):
			neighbors = self.graph[self.controllers[controller_index]]
			sorted_neighbors = sorted((weight['weight'], node) for (node, weight) in neighbors.items())
			if(new_controller_neighbor[controller_index] < len(sorted_neighbors)):
				self.controllers[controller_index] = sorted_neighbors[new_controller_neighbor[controller_index]][1]
			#if(action[controller_index] is 1):
			#	neighbors = self.graph.neighbors(self.controllers[controller_index])
			#	neigh = list(neighbors)
			#	choice = np.random.choice(neigh)
			#	while(controller_index != self.graph.nodes[choice]['cluster']):
			#		choice = np.random.choice(neigh)
			#	self.controllers[controller_index] = choice
		#Construct the state (boolean array of size <number of switches> indicating whether a switch is a controller)
		state = np.zeros(shape=len(self.graph.nodes))
		state[self.controllers] = 1
		return (state, super().step(self.controllers), False, {})

	def reset(self):
		self.controllers = self.original_controllers.copy()
		state = np.zeros(shape=len(self.graph.nodes))
		state[self.controllers] = 1
		return state

	def calculateOptimal(self):
		combinations = list(itertools.product(*self.clusters))
		min_dist = 1000000
		min_combination = None
		for combination in combinations:
			dist = self.stepLA(self.graph, combination)
			if(dist < min_dist):
				min_dist = dist
				min_combination = combination
		return (min_combination, min_dist)