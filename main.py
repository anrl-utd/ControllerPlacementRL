"""
Main code to create graph and run agent
"""
import gym
import numpy as np
import controller_env
from controller_env.envs.graph_env import generateGraph
import random
import matplotlib.pyplot as plt
import math

graph, clusters, pos = generateGraph(5, 100, draw=True)	#Generate graph
##Nudging environment
loops = 0
env = gym.make('Controller-RandomPlacement-v0', graph=graph, clusters=clusters, pos=pos)
while loops < 100:
	(obs, reward) = env.step([np.random.randint(0, 2) for i in clusters])	#This throws errors every so often since it selects neighbors without distinction whether the neighbor is in the cluster or not
	print("REWARD: ", reward)
	loops += 1


##Base environment
##env = gym.make('Controller-v0', graph=graph, clusters=clusters, pos=pos)
#env.render()
#env.reset()