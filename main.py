"""
Main code to create graph and run agent
"""
import gym
import numpy as np
import controller_env
from controller_env.envs.graph_env import generateGraph

graph, clusters, pos = generateGraph(3, 27, draw=False)	#Generate graph

#Nudging environment
env = gym.make('Controller-RandomPlacement-v0', graph=graph, clusters=clusters, pos=pos)
reward = env.step([np.random.randint(0, 2) for i in clusters])	#This throws errors every so often since it selects neighbors without distinction whether the neighbor is in the cluster or not

#Base environment
#env = gym.make('Controller-v0', graph=graph, clusters=clusters, pos=pos)
#action = [np.random.choice(i) for i in clusters]
#print(action)
#reward = env.step(action)
print("REWARD:", reward)
env.render()
env.reset()