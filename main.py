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

from stable_baselines import PPO1

if __name__ == "__main__":
	graph, clusters, pos = generateGraph(3, 45, draw=False)	#Generate graph
	#Nudging environment
	env = gym.make('Controller-RandomPlacement-v0', graph=graph, clusters=clusters, pos=pos)
	#act = []
	#for i in range(3):
	#	act.append([5 - i for i in range(env.degree)])
	#env.render()
	#env.step(act)
	#Agent
	model = PPO1('MlpPolicy', env, tensorboard_log='train_log', verbose=1)
	# Train the agent
	model.learn(total_timesteps=int(1e7))

	loops = 0
	obs = env.reset()
	while loops < 1000:
		action, states = model.predict(obs)
		(obs, reward, _, _) = env.step(action)
		print("CONTROLLERS:", env.controllers)
		print("OBSERVATION:", obs)
		print("REWARD: ", reward)
		loops += 1
	(best_controllers, best_dist) = env.calculateOptimal()
	print("Optimal:", best_controllers)
	print("Optimal Distance:", best_dist)
	env.render()


##Base environment
##env = gym.make('Controller-v0', graph=graph, clusters=clusters, pos=pos)
#env.render()
#env.reset()