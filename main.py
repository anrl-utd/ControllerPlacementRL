"""
Main code to create graph and run agent
"""
import gym
import time
import numpy as np
import controller_env
from controller_env.envs.graph_remove import generateGraph
import random

import matplotlib.pyplot as plt
import math
import networkx as nx
random.seed(3)
# from stable_baselines import PPO2
import optuna

# def optimize_algorithm(trial):
# 	"""Optimizes an algorithm using Optuna (tries out different parameters)"""
# 	#TODO: Ensure early pruning of trials occurs to speed up optimization (Tensorflow hook?)
#
# 	model_params = {
# 		'gamma': trial.suggest_loguniform('gamma', 0.9, 0.999),
# 		# 'adam_epsilon': trial.suggest_uniform('adam_epsilon', 1e-8, 1e-4),
# 		'learning_rate': trial.suggest_uniform('learning_rate', 0, 0.15),
# 		'lam': trial.suggest_uniform('lam', 0.9, 1),
# 		'cliprange': trial.suggest_uniform('cliprange', 0.1, 0.6)
# 	}
#
# 	graph, clusters, pos = generateGraph(3, 45, draw=False)	#Generate graph
# 	#Nudging environment
# 	env = gym.make('Controller-RandomPlacement-v0', graph=graph, clusters=clusters, pos=pos)
# 	#Agent
# 	model = PPO2('MlpPolicy', env, tensorboard_log='train_log', verbose=0, **model_params)
# 	# Train the agent
# 	model.learn(total_timesteps=int(1e4))
#
# 	loops = 0
# 	obs = env.reset()
# 	reward = 0 #We want the last reward to be minimal (perhaps instead do cumulative?)
# 	while loops < 100:
# 		action, states = model.predict(obs)
# 		(obs, reward, _, _) = env.step(action)
# 		loops += 1
# 	trial.report(reward)
# 	return reward #Optuna by default minimizes, so changing this to positive distance
#
# if __name__ == "__main__":
# 	#I store the results in a SQLite database so that it can resume from checkpoints.
# 	study = optuna.create_study(study_name='ppo_nudging', storage='sqlite:///params.db', load_if_exists=True)
# 	study.optimize(optimize_algorithm, n_trials=1000)

#Training without Optuna, so that we can compare the trained model to best possible contrllers
if __name__ == "__main__":
	num_correct = 0
	total_num = 0
	best_times = []
	optimal_times = []
	for x in range(1, 100):
		np.random.seed(x)
		graph, clusters, bridge_nodes, pos = generateGraph(9, 180, draw=False)  # Generate graph
		# Nudging environment
		env = gym.make('ControllerRemoveNodes-v0', graph=graph, clusters=clusters, bridge_nodes=bridge_nodes, pos=pos)
		print("Done")
		t0 = time.time()
		best_action, best_distance, best_action_not_SA, best_distance_not_SA, SA_time= env.best_action()
		t1 = time.time()
		best_action_time = t1-t0
		# t0 = time.time()
		# optimal_action = env.calculateOptimal()
		# t1 = time.time()
		# optimal_action_time = t1-t0
		print("Best action is; ", best_action)
		print("Time is ", best_action_time)
		print("Best action distance is: ", best_distance)
		print("the Simulated annealing time is: ", SA_time)
		print("Best action (not SA) is:", best_action_not_SA)
		print("Best action (not SA) distance is:", best_distance_not_SA)
		# print("Algorithm action is:", optimal_action)
		# print("Time is ", optimal_action_time)
		best_times.append(best_action_time)
		# optimal_times.append(optimal_action_time)
		if best_distance < best_distance_not_SA:
			num_correct += 1
		total_num +=1
	print("Improvement = :", num_correct / total_num)

	# print("Best times are: " , best_times)
	print("Optimal times are:", optimal_times)
	# plt.plot(best_times)
	# plt.plot(optimal_times)
	# plt.show()
	# graph, clusters, bridge_nodes, pos = generateGraph(4 , 24, draw=False)	#Generate graph
	# #Nudging environment
	# print(gym.envs.registry.all())
	# env = gym.make('ControllerRemoveNodes-v0', graph=graph, clusters=clusters, bridge_nodes = bridge_nodes, pos=pos)
	#
	# print(env.findGraphCentroid())
	# # env.repeatBestAction()
	# # env.render()
	# # env.showSubgraph(1)
	# # env.minimumSpanningTree()
	# # env.render()
	# print("This is new propagation optimal")
	# t0 = time.time()
	#
	# answer = env.calculateNewOptimal()
	# t1 = time.time()
	# print("Answer is:", answer)
	# print("The time was", t1-t0)
	# print("This is the new graph rendering!")
	# print(type(answer))
	# env.renderCentroid(answer)
	# print("This is path to centroid optimal")
	# t0 = time.time()
	# answer = env.best_action()
	# answer.sort()
	# t1 = time.time()
	# print("Answer is:", answer)
	# print("The time was", t1 - t0)
	#
	# # env.calculateDistance(answer)
	# # print(env.calculateDistance([0, 5, 8]))
	# # print(env.calculateDistance([0, 3, 8]))
	# # print(env.calculateDistance([5, 31, 41]))
	# t0 = time.time()
	# print("For this class, the optimal is", env.calculateOptimal())
	# t1 = time.time()
	# print("The time this took was", t1-t0)\
	#
	#
	# # print("WIthoug changing graph", env.calculateDistance([3, 8, 12, 15, 21]))
	# env.reset()
	# for x in range(1000):
	# 	env.recalculateBridgeWeight()
	# 	# env.render()
	# 	done = env.step()
	# 	if done == True:
	# 		break;

	# env.render()
	# print("Best controllers are:", env.showRemainingControllers())
	# print(env.calculateDistance([0, 18, 36]))

	# env2 = gym.make('Controller-RandomPlacement-v0', graph=graph, clusters=clusters, pos=pos)
	# t0 = time.time()
	# print(env2.calculateOptimal())
	# t1 = time.time()
	# print("This time is: ", t1-t0)

	# print("WIth changing  graph", env.calculateDistance([3, 8, 10, 15, 20]))

#act = []
	#for i in range(3):
	#	act.append([5 - i for i in range(env.degree)])
	#env.render()
	#env.step(act)
	#Agent
	# model = PPO2('MlpPolicy', env, tensorboard_log='train_log', verbose=1)
	# Train the agent
	# model.learn(total_timesteps=int(1))

	# loops = 0
	# obs = env.reset()
	# while loops < 1000:
	# 	action, states = env.predict(obs)
	# 	(obs, reward, _, _) = env.step(action)
	# 	# print("CONTROLLERS:", env.controllers)
	# 	# print("OBSERVATION:", obs)
	# 	# print("REWARD: ", reward)
	# 	loops += 1
	# (best_controllers, best_dist) = env.calculateOptimal()
	# print("Optimal:", best_controllers)
	# print("Optimal Distance:", best_dist)
# 	env.render()


# Testing best number of packets per environment step (to evaluate controller placement)
# print("Starting trials . . .")
# for i in range(10):
# 	n_clusters = i + 2
# 	n_nodes = (i + 2) * np.random.randint(5, 10)
# 	graph, clusters, pos = generateGraph(n_clusters, n_nodes, draw=False)	#Generate graph
# 	env = gym.make('Controller-v0', graph=graph, clusters=clusters, pos=pos)
# 	controllers = env._random_valid_controllers()
# 	num_trials = 1000
# 	num_steps = 20
# 	trial_data = np.zeros(shape=(num_trials, num_steps))
# 	for trial in range(num_trials):
# 		for step in range(num_steps):
# 			trial_data[trial, step] = env.step(controllers, num_packets=10 + step * 20) / (10 + step * 20.0)
# 	print(n_clusters, n_nodes)
# 	print(np.mean(trial_data, axis=0))
# 	print(np.std(trial_data, axis=0))