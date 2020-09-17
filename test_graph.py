import networkx as nx
import numpy as np
from controller_env.envs.graph_env import generateClusters, ControllerEnv
import pickle

#k_graph = nx.fast_gnp_random_graph(95, 0.05)
#while(not nx.is_connected(k_graph)):
#	k_graph = nx.fast_gnp_random_graph(95, 0.05)
#k_graph = nx.graphml.read_graphml('Uninett2010.graphml')
#edge_dict = nx.get_edge_attributes(k_graph, 'LinkSpeed')
#new_edges = { key: float(value) for key, value in edge_dict.items() }
#nx.set_edge_attributes(k_graph, new_edges, 'weight')
#print(nx.get_edge_attributes(k_graph, 'weight'))
#graph, clusters, pos = generateClusters(k_graph)
#print(list(graph.nodes.data()))
#print(clusters)

clusters = pickle.load(open('clusters.pickle', 'rb'))
graph = nx.read_gpickle('graph.gpickle')
env = ControllerEnv(graph, clusters)
print(env.graphCentroidAction())