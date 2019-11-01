"""
Main code to create graph and run agent
"""
import gym
import numpy as np
import controller_env
from controller_env.envs.graph_env import generateGraph

graph = generateGraph(3, 12, draw=True)
env = gym.make('Controller-v0', graph=graph)
env.step(0)
env.reset()
env.render()