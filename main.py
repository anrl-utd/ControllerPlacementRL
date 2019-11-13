"""
Main code to create graph and run agent
"""
import gym
import numpy as np
import controller_env
from controller_env.envs.graph_env import generateGraph

graph, clusters = generateGraph(3, 27, draw=False)
env = gym.make('Controller-v0', graph=graph, clusters=clusters)

#Choose random controller for each cluster
reward = env.step([np.random.choice(i) for i in clusters])
print("REWARD:", reward)
env.render()
env.reset()