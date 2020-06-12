import numpy as np
from gym import spaces
import itertools
import networkx as nx
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
		# self.action_space = spaces.Box(0, 1, (len(graph.nodes),), dtype=np.float32)
		self.observation_space = spaces.Box(0, 1, (len(graph.nodes),), dtype=np.bool)
		self.controllers = []
		self.best_controllers = []
		self.best_reward = 100000

		self.num_clusters = len(clusters)
		print(self.num_clusters)

		# Created to speed up getting cluster info since graph is static
		cluster_info = nx.get_node_attributes(self.graph, 'cluster')
		self.cluster_info = np.array(list(cluster_info.items()), dtype=np.int32)  # Construct numpy array as [[node num, cluster num], [..]]

		# Created to speed up creating observation
		self.state = np.zeros(shape=len(self.graph.nodes))

	def step(self, action):
		"""
		Steps environment once.
		Args:
			action: Action to perform
					Index of node to set as controller
		Returns:
			Tuple of (State, Reward) after selecting controllers and passing to base environment (latency for 1000 packets)
			State is a boolean array of size <number of switches> which indicates whether a switch is a controller or not
		"""
		self.controllers.append(action)
		# Construct the state (boolean array of size <number of switches> indicating whether a switch is a controller)
		action_cluster = self.cluster_info[self.cluster_info[:, 0] == action][0][1]  # Get the cluster the action is part of
		
		self.state[(self.cluster_info[:, 1] == action_cluster) & (self.state != 1)] = -1  # Set all nodes of same cluster except controllers as -1
		self.state[action] = 1  # Set new controller as 1

		# TODO: Speed up by using self.state == -1 to determine if the action is erroneous and just set reward of -10000 (no need to do controller setting)
		(obs, rew, done, i) = (self.state.copy(), super().step(self.controllers), len(self.controllers) >= self.num_clusters, {})
		"""
		if rew >= 10000:
			done = True
		"""
		if done:
			if self.best_reward > rew:
				self.best_controllers = self.controllers
				self.best_reward = rew
		return (obs, -rew, done, i)

	def reset(self):
		"""Resets environment"""
		self.controllers = []
		self.state = np.zeros(shape=len(self.graph.nodes))
		return self.state.copy()

	def calculateOptimal(self) -> (list, int):
		"""
		Override of the calculateOptimal() in the base environment class
		Returns:
			Best combination of controllers
			Total distance between return controllers
		"""
		combinations = list(itertools.product(*self.clusters))
		min_dist = 1000000
		min_combination = None
		for combination in combinations:
			dist = super().step(combination)
			if(dist < min_dist):
				min_dist = dist
				min_combination = combination
		return (min_combination, min_dist)

	def optimal_neighbors(self, graph, controllers : list) -> (list, int):
		"""
		Gets best set of controllers from neighbors of provided set of nodes
		Args:
			graph (nx.Graph): NetworkX graph to use
			controllers: List of node numbers as controllers
		Returns:
			Best combination of controllers
			Total distance between return controllers
		"""
		# This isn't efficient and does not take advantage of other variables in the class
		# TODO: Optimize to use cluster_info
		clusters = nx.get_node_attributes(graph, 'cluster')
		neighbors_list = []
		for i in controllers:
			cluster = []
			cluster.append(i)
			neighbors = graph.neighbors(i)
			for neighbor in neighbors:
				if(clusters[neighbor] == clusters[i]):
					cluster.append(neighbor)
			neighbors_list.append(cluster)
		print(neighbors_list)
		# Find best controller set from neighbors
		combinations = list(itertools.product(*neighbors_list))
		min_dist = 1000000
		min_combination = None
		for combination in combinations:
			dist = super().step(combination)
			if(dist < min_dist):
				min_dist = dist
				min_combination = combination
		return (min_combination, min_dist)