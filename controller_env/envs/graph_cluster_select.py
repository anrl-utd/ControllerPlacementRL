import numpy as np
from gym import spaces
import networkx as nx
import itertools
from controller_env.envs.graph_env import ControllerEnv

class ControllerClusterSelect(ControllerEnv):
	"""
	Environment that selects clusters one-at-a-time. Compared to ControllerSlowSelect,
	the action space is 0-K where K is the max nodes per cluster. The model picks which 
	node in a cluster every step (rather than which node in entire graph) so no chance for
	bad controllers
	Input: Graph with clusters
	Action: List of max cluster size with probabilities for each neighbor to be controller.
			Argmax across lists to choose index of neighbor to nudge controller to
	State: List of whether each node is controller or not [0, 1, ..., 0]
	Reward: ControllerEnv reward
	"""
	def __init__(self, graph, clusters, pos=None):
		"""Initilizes environment and assigns random nodes to be controllers"""
		super().__init__(graph, clusters, pos, check_controller_num=False)
		self.average_graph = self.graph.copy()
		self.num_resets = 0
		self.controllers = []
		self.best_controllers = []
		self.best_reward = 100000
		self.cumulative_reward = 0  # Cumulative reward since env init - does not reset

		self.num_clusters = len(clusters)
		print(self.num_clusters)

		self.original_graph = graph.copy()

		# Created to speed up getting cluster info since graph is static
		cluster_info = nx.get_node_attributes(self.graph, 'cluster')
		self.cluster_info = np.array(list(cluster_info.items()), dtype=np.int32)  # Construct numpy array as [[node num, cluster num], [..]]
		self.max_cluster_len = max(len(cluster) for cluster in clusters)
		self.cluster_index = 0
		self.clusters = clusters
		self.action_space = spaces.Discrete(self.max_cluster_len)
		#self.action_space = spaces.Box(0, self.max_cluster_len, (2,))
		#self.observation_space = spaces.Box(-1, self.max_cluster_len, (len(clusters),))
		self.observation_space = spaces.Box(0, 1, (len(graph.nodes),), dtype=np.int32)

		# Created to speed up creating observation
		self.state = np.zeros(shape=len(self.graph.nodes))
		#self.state = np.ones(shape=len(self.clusters)) * -1  # List of -1

	def step(self, action):
		"""
		Steps environment once.
		Args:
			action: Action to perform
					Index of node to set as controller in range (0-<max cluster len>)
		Returns:
			Tuple of (State, Reward) after selecting controllers and passing to base environment (latency for 1000 packets)
			State is a boolean array of size <number of switches> which indicates whether a switch is a controller or not
		"""
		cluster_len = len(self.clusters[self.cluster_index])
		if action >= cluster_len:
			if self.best_reward > 10000:
				self.best_controllers = self.controllers
				self.best_reward = 10000
			return (self.state.copy(), -10000, True, {})

		node = self.clusters[self.cluster_index][action]
		self.controllers.append(node)
		## Construct the state (boolean array of size <number of switches> indicating whether a switch is a controller)
		#action_cluster = self.cluster_info[self.cluster_info[:, 0] == node][0][1]  # Get the cluster the action is part of
		
		#self.state[(self.cluster_info[:, 1] == action_cluster) & (self.state != 1)] = -1  # Set all nodes of same cluster except controllers as -1
		self.state[node] = 1  # Set new controller as 1
		#self.state[self.cluster_index] = 1  # action
		self.cluster_index += 1

		# TODO: Speed up by using self.state == -1 to determine if the action is erroneous and just set reward of -10000 (no need to do controller setting)
		(obs, rew, done, i) = (self.state.copy(), super().step(self.controllers), self.cluster_index >= self.num_clusters, {})
		if done:
			self.cumulative_reward += rew  # Add cumulative only when done
			if self.best_reward > rew:
				self.best_controllers = self.controllers
				self.best_reward = rew
		#else:  # Test if providing reward at end of episode is better
		#	rew = 0
		return (obs, -rew, done, i)

	def reset(self, adjust=True, full=False):
		"""Resets environment"""
		self.controllers = []
		self.state = np.zeros(shape=len(self.graph.nodes))
		#self.state = np.ones(shape=len(self.clusters)) * -1  # List of -1
		self.cluster_index = 0
		super().reset()
		if adjust:
			## Shift the optimal by some amount by doing a random increase to a path between 2 random nodes
			#start_cluster = np.random.randint(0, self.num_clusters)
			#end_cluster = np.random.randint(0, self.num_clusters)
			#while end_cluster == start_cluster:
			#	end_cluster = np.random.randint(0, self.num_clusters)
			#start_controller = np.random.choice(self.clusters[start_cluster])
			#end_controller = np.random.choice(self.clusters[end_cluster])
			##while end_controller == start_controller:
			##	end_controller = np.random.randint(0, len(self.graph.nodes))
			#path = nx.shortest_path(self.graph, source=start_controller, target=end_controller, weight='weight')
			#prior_node = start_controller
			#random_change = np.random.randint(-5, 6)
			#for i in range(1, len(path)):
			#	#print(i, self.graph[prior_node][path[i]]['weight'])
			#	self.graph[prior_node][path[i]]['weight'] += random_change
			#	#print(i, self.graph[prior_node][path[i]]['weight'])
			#	if self.graph.get_edge_data(prior_node, path[i])['weight'] < 0:
			#		self.graph[prior_node][path[i]]['weight'] = 0
			#	prior_node = path[i]
			# Modify every edge to be chosen from random distribution around other edge
			for u, v, dist in self.original_graph.edges.data('weight'):
				self.graph[u][v]['weight'] = np.random.normal(dist, 5) # (0.5 + np.random.beta(0.5, 0.5)) * dist
				self.average_graph[u][v]['weight'] = ((self.average_graph[u][v]['weight'] * self.num_resets) + self.graph[u][v]['weight']) / (self.num_resets + 1)
			self.num_resets += 1
		else:
			if full:
				self.graph = self.original_graph.copy()
		return self.state.copy()

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

	def compute_greedy_heuristic(self):
		"""
		Computes WMSCP Greedy Heuristic. self.graph is the NetworkX graph. self.clusters is a list of lists of node incides 
		where each list corresponds to the list of nodes in each cluster (so [[0,1,2,3],[4,5,6,7]] would be two clusters with 0-3 
		in one cluster and 4-7 in other).
		"""
		controller_set = []  # Stores controller indices
		distances = nx.floyd_warshall(self.graph, 'weight')  # Stores distances between every node
		weights_index = 0  # Converts index of minimum weight to actual node index (only works because clusters are in-order and sorted
		for i in range(len(self.clusters)):
			# Go through each node in cluster and compute weights
			weights = []
			for node in self.clusters[i]:
				# Go through every cluster and find shortest distance, shortest distance to controller
				weight = 0  # Stores weight of this node
				for j in range(len(self.clusters)):
					if i == j:  # Matching cluster, ignore
						continue
					cluster_distance = 1e6
					controller_distance = 0
					for cluster_node in self.clusters[j]:
						# Go through every node in other clusters and compute distance to find shortest node per cluster
						cluster_distance = min(cluster_distance, distances[node][cluster_node])
						if cluster_node in controller_set:
							# If node is a controller, add its distance a "second" time to correspond 
							# to second sigma-sum of weight calculation
							assert controller_distance == 0, print(cluster_node, controller_set)
							controller_distance = distances[node][cluster_node]
					assert cluster_distance != 1e6, "Cluster distance was too low, it was not modified"
					weight += cluster_distance + controller_distance  # Add to node weight
					weight /= len(self.clusters[i]) + 1  # Divide by nodes covered (the length of cluster upcoming controller is in)
				weights.append(weight)
			controller_set.append(weights.index(min(weights)) + weights_index)  # Add node index to controller list
			weights_index += len(self.clusters[i])
		return controller_set, super().step(controller_set)  # Return controller list and corresponding cost (minimum distance between all controllers)


	def rewardStep(self, controllers):
		# This is just there so SelectModified can call on the base class
		return super().step(controllers)

class ControllerClusterSelectModified(ControllerClusterSelect):
	def __init__(self, graph, clusters, pos=None, new_state_space=True, neg1_state_space=False, new_state_space_1=False):
		super().__init__(graph, clusters, pos)
		self.new_state_space = new_state_space
		self.neg1_state_space = neg1_state_space
		self.new_state_space_1 = new_state_space_1
		self.action_space = spaces.Discrete(self.max_cluster_len)
		if new_state_space:
			self.observation_space = spaces.Box(-1 if self.neg1_state_space else 0, self.max_cluster_len, (len(clusters),))
			self.state = np.ones(shape=len(self.clusters)) * (-1 if self.neg1_state_space else 0)

	def step(self, action):
		cluster_len = len(self.clusters[self.cluster_index])
		if action >= cluster_len:
			if self.best_reward > 10000:
				self.best_controllers = self.controllers
				self.best_reward = 10000
			return (self.state.copy(), -10000, True, {})

		node = self.clusters[self.cluster_index][action]
		self.controllers.append(node)

		if not self.new_state_space:
			self.state[node] = 1  # Set new controller as 1
			if self.neg1_state_space:
				action_cluster = self.cluster_info[self.cluster_info[:, 0] == node][0][1]  # Get the cluster the action is part of
				self.state[(self.cluster_info[:, 1] == action_cluster) & (self.state != 1)] = -1  # Set all nodes of same cluster except controllers as -1
		else:
			self.state[self.cluster_index] = (1 if self.new_state_space_1 else action)  # action
		self.cluster_index += 1

		# TODO: Speed up by using self.state == -1 to determine if the action is erroneous and just set reward of -10000 (no need to do controller setting)
		(obs, rew, done, i) = (self.state.copy(), super().rewardStep(self.controllers), self.cluster_index >= self.num_clusters, {})
		if done:
			if self.best_reward > rew:
				self.best_controllers = self.controllers
				self.best_reward = rew
		return (obs, -rew, done, i)

	def reset(self):
		ret_state = super().reset()
		self.state = np.ones(shape=len(self.clusters)) * (-1 if self.neg1_state_space else 0) if self.new_state_space else ret_state
		return self.state.copy()