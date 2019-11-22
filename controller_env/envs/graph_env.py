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
warnings.filterwarnings("ignore", category=UserWarning)

class ControllerEnv(gym.Env):
	"""Environment used to simulate the network for the RL"""
	metadata = {'render.modes' : ['human']}
	def __init__(self, graph, clusters):
		"""Initializes the environment (Runs at first)"""
		print("Initialized environment!")
		self.original_graph = graph.copy()
		self.clusters = np.stack(clusters)
		self.graph = graph.copy()

	def step(self, action):
		"""Steps the environment once"""
		print("Environment step")
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
		#Create metagraph of controllers. The node at an index corresponds to the controller of the cluster of that index
		controller_graph = self._set_controllers(action)

		#Create "packets" with source, destination
		packets = np.random.randint(low=0, high=len(self.graph.nodes), size=(100, 2))
		#Convert source and destination to cluster the source/destination is in
		distance = 0
		for i in range(packets.shape[0]):
			if packets[i, 0] == packets[i, 1]:
				continue
			source_cluster = np.where(self.clusters == packets[i, 0])[0][0]
			destination_cluster = np.where(self.clusters == packets[i, 1])[0][0]
			distance += nx.dijkstra_path_length(controller_graph, source_cluster, destination_cluster)
		return distance

	def reset(self):
		"""Resets the environment to initial state"""
		print("Reset environment")

	def render(self, mode='human', close=False):
		"""Renders the environment once"""
		print("Rendered enviroment")
		plt.clf()	# Clear the matplotlib figure

		#Redraw the entire graph (this can only be expedited if we save the position and colors beforehand)
		#Then we won't have to recalculate all this to draw. Maybe make them a global variable?
		pos = nx.spring_layout(self.graph)
		node_colors = np.arange(0, nx.number_of_nodes(self.graph), 1)
		clustering = nx.get_node_attributes(self.graph, 'cluster')
		for node in clustering:
			node_colors[node] = clustering[node]
		nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors)
		nx.draw_networkx_edges(self.graph, pos, self.graph.edges())        # draw the edges of the self.graph
		nx.draw_networkx_labels(self.graph, pos)                      # draw  the labels of the self.graph
		edge_labels = nx.get_edge_attributes(self.graph,'weight')
		nx.draw_networkx_edge_labels(self.graph,pos,edge_labels=edge_labels) # draw the edge weights of the self.graph
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
		new_contr_indices = []
		for i in range(len(controllers)):
			new_contr_indices.append([i, controllers[i]])
		controller_graph = nx.complete_graph(len(new_contr_indices))	#Store controller metagraph
		for pair in itertools.combinations(new_contr_indices, 2):
			controller_graph.add_edge(pair[0][0], pair[1][0], weight=nx.dijkstra_path_length(self.graph, source=pair[0][1], target=pair[1][1]))

		#Display metagraph for debugging. Should be removed once we get _set_controllers() working
		pos = nx.spring_layout(controller_graph)
		nx.draw_networkx_nodes(controller_graph, pos)
		nx.draw_networkx_edges(controller_graph, pos, controller_graph.edges())        # draw the edges of the controller_graph
		nx.draw_networkx_labels(controller_graph, pos)                      # draw  the labels of the controller_graph
		edge_labels = nx.get_edge_attributes(controller_graph,'weight')
		nx.draw_networkx_edge_labels(controller_graph,pos,edge_labels=edge_labels) # draw the edge weights of the controller_graph
		plt.draw()
		plt.show()

		return controller_graph

class ControllerRandomStart(ControllerEnv):
	def __init__(self, graph, clusters):
		super().__init__(graph, clusters)
		self.controllers = [np.random.choice(i) for i in self.clusters]

	def step(self, action):
		for controller_index in range(len(action)):
			if(action[controller_index] is 1):
				neighbors = self.graph.neighbors(self.controllers[controller_index])
				self.controllers[controller_index] = np.random.choice(list(neighbors))
		return super().step(self.controllers)

def generateGraph(num_clusters, num_nodes, prob=0.2, weight_low=0, weight_high=100, draw=True):
	"""Generates graph given number of clusters and nodes
	Args:
		num_clusters: Number of clusters
		num_nodes: Number of nodes
		prob: Parameter for graph (probability of triangle after adding edge)
		weight_low: Lowest possible weight for edge in graph
		weight_high: Highest possible weight for edge in graph
		draw: Whether or not to show graph (True indicates to show)
	
	Returns:
		Graph with nodes in clusters, array of clusters
	"""
	graph = nx.powerlaw_cluster_graph(num_nodes,3, prob, random.seed(a = None, version = 2))

	traversal = list(nx.bfs_tree(graph, source = 0))    # get a bft of the graph in a list
	array_traversal = np.array_split(traversal, num_clusters)      # split the bft list into equal parts
	pos = nx.spring_layout(graph)   # get the positions of the nodes of the graph

	cluster_attrib = dict() # Dictionary that stores cluster number for each node
	node_colors = np.arange(0, num_nodes, 1, np.uint8)
	# Assign cluster attribute for each node
	for node in graph.nodes:
		for index, the_traversal in enumerate(array_traversal): # Get clusters and find cluster node is in
			if node in the_traversal:
				cluster_attrib[node] = index
				node_colors[node] = index
	# Set node cluster numbers and draw them
	nx.set_node_attributes(graph, cluster_attrib, 'cluster')
	nx.draw_networkx_nodes(graph, pos, node_color=node_colors)

	# Keeping Vincent's code in case we decide to have a legend for the cluster lables [Usaid]
	#for index, the_traversal in enumerate(array_traversal): # for each cluster and index of the graph
	#	subpos = dict()                                # create a sub dictionary of positions for each cluster's ndes
	#	for key, value in pos.items():
	#		if key in the_traversal:
	#			subpos[key] = value
	#	randColor = "#" + str(random.randint(0, 999999)).zfill(6)   # get a random hexadecimal colour
	#	if draw:
	#		nx.draw_networkx_nodes(graph.subgraph(the_traversal), subpos, node_color=randColor, label = "Cluster " + str(index))  # draw an individual cluster
	
	#Assign distance weights to edges
	for edge in graph.edges:
		graph.add_edge(edge[0], edge[1], weight=random.randint(weight_low, weight_high))
	
	if draw:
		nx.draw_networkx_edges(graph, pos, graph.edges())        # draw the edges of the graph
		nx.draw_networkx_labels(graph, pos)                      # draw  the labels of the graph
		edge_labels = nx.get_edge_attributes(graph,'weight')
		nx.draw_networkx_edge_labels(graph,pos,edge_labels=edge_labels) # draw the edge weights of the graph
		#plt.savefig("path.png")
		plt.draw()
		plt.legend()
		plt.show()
	return graph, array_traversal
