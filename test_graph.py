import networkx as nx
import gym
import numpy as np
import controller_env
import pickle

clusters = pickle.load(open('C:/Users/usaid/Downloads/clusters.pickle', 'rb'))
graph = nx.read_gpickle('C:/Users/usaid/Downloads/graph.gpickle')
env = gym.make('Controller-v0', graph=graph, clusters=clusters)
env.reset()
env.render()
for i in range(15):
	print(env.step([13, 30, 68, 55, 79, 15 + i]))
print(env.step([9, 29, 30, 55, 79, 68]))