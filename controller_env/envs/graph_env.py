"""
Environment for reinforcement learning
"""
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import warnings
import random
import itertools
import pprint
from collections import defaultdict
warnings.filterwarnings("ignore", category=UserWarning)

class ControllerEnv(gym.Env):
	"""Base environment used to simulate the network for the RL"""
	metadata = {'render.modes' : ['human']}
	def __init__(self, graph, clusters, pos=None):
		"""Initializes the environment (Runs at first)"""
		print("Initialized environment!")
		self.action_space = spaces.Box(np.zeros(len(clusters)), np.ones(len(clusters)) * len(graph.nodes), dtype=np.uint8)
		#self.observation_space = spaces.Box(np.zeros(shape=len(graph.nodes)), np.ones(shape=len(graph.nodes)), dtype=np.bool)
		self.original_graph = graph.copy()
		if(pos is None):
			self.pos = nx.spring_layout(graph)   # get the positions of the nodes of the graph
		else:
			self.pos = pos
		self.clusters = np.stack(clusters)
		self.graph = graph.copy()
		self.degree = self._graph_degree()

	def step(self, action, num_packets=100):
		"""Steps the environment once"""
		"""
		How it works:
		action is indices of controllers
		Create a new complete graph with controllers only
		 - Use shortest-distance between controllers as the weight of edges for controller graph
		 - Store indices of nodes under each controller
		Create several "packets" with source and destination
		Find controller for source and destination
		If source is same as destination controller, distance is 0
		Otherwise, distance is the shortest-path distance between the source controller and destination
		Add up all distances and have that as initial reward
		"""
		distance = 0
		controller_graph = None
		#Create metagraph of controllers. The node at an index corresponds to the controller of the cluster of that index
		try:
			controller_graph = self._set_controllers(action)
			"""
			Don't create packets, try having reward of total distance between all adjacent controllers

			#Create "packets" with source, destination
			packets = np.random.randint(low=0, high=len(self.graph.nodes), size=(num_packets, 2))
			#Convert source and destination to cluster the source/destination is in
			for i in range(packets.shape[0]):
				if packets[i, 0] == packets[i, 1]:
					continue
				source_cluster = np.where(self.clusters == packets[i, 0])[0][0]
				destination_cluster = np.where(self.clusters == packets[i, 1])[0][0]
				distance += nx.dijkstra_path_length(controller_graph, source_cluster, destination_cluster)
			"""
		except AssertionError:
			return 100000
		#Return output reward
		#return -distance
		return controller_graph.size(weight='weight')

	def reset(self):
		"""Resets the environment to initial state"""
		print("Reset environment")

	def render(self, mode='human', close=False):
		"""Renders the environment once"""
		plt.clf()	# Clear the matplotlib figure

		#Redraw the entire graph (this can only be expedited if we save the position and colors beforehand)
		#Then we won't have to recalculate all this to draw. Maybe make them a global variable?
		node_colors = np.arange(0, nx.number_of_nodes(self.graph), 1)
		clustering = nx.get_node_attributes(self.graph, 'cluster')
		for node in clustering:
			node_colors[node] = clustering[node]
		nx.draw_networkx_nodes(self.graph, self.pos, node_color=node_colors)
		nx.draw_networkx_edges(self.graph, self.pos, self.graph.edges())        # draw the edges of the self.graph
		nx.draw_networkx_labels(self.graph, self.pos)                      # draw  the labels of the self.graph
		edge_labels = nx.get_edge_attributes(self.graph,'weight')
		nx.draw_networkx_edge_labels(self.graph,self.pos,edge_labels=edge_labels) # draw the edge weights of the self.graph
		plt.draw()
		plt.show()

	def _set_controllers(self, controllers):
		"""Creates metagraph of controllers
		Args:
			controllers: Array of controller indices
		
		Returns:
			Complete graph of controllers (metagraph)
			
		Raises:
			AssertError: Issue with controller indices (not 1 per cluster)"""
		#Ensure that these are valid controllers - all clusters have a controller
		assert(len(controllers) == self.clusters.shape[0])
		found_clusters = np.zeros((len(controllers)))	#Stores what clusters have controllers been found for
		clusters = nx.get_node_attributes(self.graph, 'cluster')
		index = 0
		for controller in controllers:
			#Multiple controllers in a cluster
			assert(found_clusters[clusters[controller]] == 0)
			found_clusters[clusters[controller]] = 1
		
		#Controllers were found to be valid. Now add controllers to complete metagraph.
		#TODO: Optimize this, new_contr_indices and mapping can be reduced to a single variable (and possible a single line for the for)
		new_contr_indices = []
		#mapping = defaultdict(list)
		for i in range(len(controllers)):
			new_contr_indices.append([i, controllers[i]])
			#mapping[i] = controllers[i]
		controller_graph = nx.complete_graph(len(new_contr_indices))	#Store controller metagraph
		
		for pair in itertools.combinations(new_contr_indices, 2):
			controller_graph.add_edge(pair[0][0], pair[1][0], weight=nx.dijkstra_path_length(self.graph, source=pair[0][1], target=pair[1][1]))

		##Display metagraph for debugging. Should be removed once we get _set_controllers() working
		#display_graph = nx.relabel_nodes(controller_graph, mapping)
		## nx.draw_networkx_nodes(display_graph,self. pos)
		#nx.draw_networkx_edges(display_graph, self.pos, display_graph.edges())        # draw the edges of the display_graph
		#nx.draw_networkx_labels(display_graph, self.pos)                      # draw  the labels of the display_graph
		#edge_labels = nx.get_edge_attributes(display_graph,'weight')
		#nx.draw_networkx_edge_labels(display_graph,self.pos,edge_labels=edge_labels) # draw the edge weights of the display_graph
		#plt.draw()
		#plt.show()

		return controller_graph

	def _graph_degree(self):
		"""Returns the highest degree of a node in the graph"""
		return max([degree for node, degree in self.graph.degree()])
			
	def _random_valid_controllers(self):
		"""Intended for testing, this gives a random set of valid controllers"""
		cluster_arr = np.asarray(self.graph.nodes.data('cluster')) #Creates NumPy array with [node #, cluster #] rows
		controller_indices = []
		for cluster in range(self.clusters.shape[0]): #For every cluster
			cluster_controller = np.random.choice(cluster_arr[cluster_arr[:, 1] == cluster][:, 0]) #Select all nodes of a cluster then choose one randomly
			controller_indices.append(cluster_controller)
		return controller_indices

	def stepLA(self, graph, controllers):
		"""
		Helper function used to calculate distance between chosen controllers
		:param controllers: The list of chosen controllers in which to calculate distance between
		:return: The total distance from every controller to the other controllers
		"""
		distance = 0
		for current_controller in range(len(controllers)):
			for other_controller in range(current_controller, len(controllers)):
				distance += nx.dijkstra_path_length(graph, controllers[current_controller],
												   controllers[other_controller])
		return distance

	def calculateOptimal(self):
		combinations = list(itertools.product(*self.clusters))
		min_dist = 1000000
		min_combination = None
		for combination in combinations:
			dist = self.step(combination)
			if(dist < min_dist):
				min_dist = dist
				min_combination = combination
		return (min_combination, min_dist)


def generateGraph(num_clusters, num_nodes, prob_cluster=0.5, prob=0.2, weight_low=0, weight_high=100, draw=True):
	"""Generates graph given number of clusters and nodes
	Args:
		num_clusters: Number of clusters
		num_nodes: Number of nodes
		prob_cluster: Probability of adding edge between any two nodes within a cluster
		prob: Probability of adding edge between any two nodes
		weight_low: Lowest possible weight for edge in graph
		weight_high: Highest possible weight for edge in graph
		draw: Whether or not to show graph (True indicates to show)
	
	Returns:
		Graph with nodes in clusters, array of clusters, graph position for drawing
	"""
	node_colors = np.arange(0, num_nodes, 1, np.uint8) #Stores color of nodes
	G = nx.Graph()
	node_num = 0
	nodes_per_cluster = int(num_nodes / num_clusters)
	clusters = np.zeros((num_clusters, nodes_per_cluster), np.uint8) #Stores nodes in each cluster

	#Create clusters and add random edges within each cluster before merging them into single graph
	for i in range(num_clusters):
		#Add tree to serve as base of cluster subgraph. Loop through all edges and assign weights to each
		cluster = nx.random_tree(nodes_per_cluster)
		for start, end in cluster.edges:
			cluster.add_edge(start, end, weight=random.randint(weight_low, weight_high))

		#Add edges to increase connectivity of cluster
		new_edges = np.random.randint(0, nodes_per_cluster, (int(nodes_per_cluster * prob_cluster), 2))
		new_weights = np.random.randint(weight_low, weight_high, (new_edges.shape[0], 1))
		new_edges = np.append(new_edges, new_weights, 1)
		cluster.add_weighted_edges_from(new_edges)

		#Set attributes and colors
		nx.set_node_attributes(cluster, i, 'cluster')
		nx.set_node_attributes(cluster, 0.5, 'learning_automation')
		node_colors[node_num:(node_num + nodes_per_cluster)] = i
		node_num += nodes_per_cluster
		clusters[i, :] = np.asarray(cluster.nodes) + nodes_per_cluster * i

		#Merge cluster with main graph
		G = nx.disjoint_union(G, cluster)

	#Add an edge to connect all clusters (to gurantee it is connected)
	node_num = 0
	for i in range(num_clusters - 1):
		G.add_edge(node_num, node_num + nodes_per_cluster, weight=random.randint(weight_low, weight_high))
		node_num += nodes_per_cluster

	#Add random edges to any nodes to increase diversity
	new_edges = np.random.randint(0, num_nodes, (int(num_nodes * 0.1), 2))
	new_weights = np.random.randint(weight_low, weight_high, (new_edges.shape[0], 1))
	new_edges = np.append(new_edges, new_weights, 1)
	G.add_weighted_edges_from(new_edges)
	G.remove_edges_from(nx.selfloop_edges(G)) #Remove self-loops caused by adding random edges

	#Draw graph
	pos = nx.spring_layout(G)
	if draw:
		nx.draw_networkx_nodes(G, pos, node_color = node_colors)
		nx.draw_networkx_labels(G, pos)
		nx.draw_networkx_edges(G, pos, G.edges())
		plt.draw()
		plt.show()
	return G, clusters, pos

