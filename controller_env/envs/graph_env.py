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
warnings.filterwarnings("ignore", category=UserWarning)


class ControllerEnv(gym.Env):
	"""Environment used to simulate the network for the RL"""
	metadata = {'render.modes' : ['human']}

	def __init__(self, graph, clusters):
		"""Initializes the environment (Runs at first)"""
		print("Initialized environment!")
		self.original_graph = graph.copy()
		self.clusters = clusters
		self.graph = graph.copy()

	def step(self, action):
		"""Steps the environment once"""
		print("Environment step")
		self._set_controllers(action)

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
		for cluster_num in range(len(self.clusters)):
			for neighbors in nx.all_neighbors(self.graph, controllers[cluster_num]):
				# TODO: Separate controllers from all nodes not in their cluster, then link all nodes in its cluster directly to it
				print(controllers[cluster_num], neighbors)

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
	print(nx.get_node_attributes(graph, 'cluster'))
	for node in nx.get_node_attributes(graph, 'cluster'):
		print(node)
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
