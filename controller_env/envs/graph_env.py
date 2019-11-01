"""
Environment for reinforcement learning
"""
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import warnings
import random
warnings.filterwarnings("ignore", category=UserWarning)


class ControllerEnv(gym.Env):
	"""Environment used to simulate the network for the RL"""
	metadata = {'render.modes' : ['human']}

	def __init__(self, graph):
		"""Initializes the environment (Runs at first)"""
		print("Initialized environment!")

	def step(self, action):
		"""Steps the environment once"""
		print("Environment step")
		
	def reset(self):
		"""Resets the environment to initial state"""
		print("Reset environment")

	def render(self, mode='human', close=False):
		"""Renders the environment once"""
		print("Rendered enviroment")

def generateGraph(numClusters, numNodes, prob=0.2, draw=True):
	"""Generates graph given number of clusters and nodes
	Args:
		numClusters: Number of clusters
		numNodes: Number of nodes
		prob: Parameter for graph (probability of triangle after adding edge)
		draw: Whether or not to show graph (True indicates to show)
	
	Returns:
		Graph with nodes in clusters
	"""
	graph = nx.powerlaw_cluster_graph(numNodes,3, prob, random.seed(a = None, version = 2))

	traversal = list(nx.bfs_tree(graph, source = 0))    # get a bft of the graph in a list
	array_traversal = np.array_split(traversal, numClusters)      # split the bft list into equal parts

	pos = nx.spring_layout(graph)   # get the positions of the nodes of the graph
	for index, the_traversal in enumerate(array_traversal): # for each cluster and index of the graph
		subpos = dict()                                # create a sub dictionary of positions for each cluster's ndes
		for key, value in pos.items():
			if key in the_traversal:
				subpos[key] = value
		randColor = "#" + str(random.randint(0, 999999)).zfill(6)   # get a random hexadecimal colour
		if draw:
			nx.draw_networkx_nodes(graph.subgraph(the_traversal), subpos, node_color=randColor, label = "Cluster " + str(index))  # draw an individual cluster
	if draw:
		nx.draw_networkx_edges(graph, pos, graph.edges())        # draw the edges of the graph
		nx.draw_networkx_labels(graph, pos)                      # draw the labels of the graph

		plt.savefig("path.png")
		plt.draw()
		plt.legend()
		plt.show()
