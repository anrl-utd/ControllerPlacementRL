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
import optuna

def optimize_algorithm(trial):
	"""Optimizes an algorithm using Optuna (tries out different parameters)"""
	#TODO: Ensure early pruning of trials occurs to speed up optimization (Tensorflow hook?)

	model_params = {
		'gamma': trial.suggest_loguniform('gamma', 0.9, 0.999),
		'adam_epsilon': trial.suggest_uniform('adam_epsilon', 1e-8, 1e-4),
		'lam': trial.suggest_uniform('lam', 0.9, 1),
		'clip_param': trial.suggest_uniform('clip', 0.1, 0.4)
	}

	graph, clusters, pos = generateGraph(3, 45, draw=False)	#Generate graph
	#Nudging environment
	env = gym.make('Controller-RandomPlacement-v0', graph=graph, clusters=clusters, pos=pos)
	#Agent
	model = PPO1('MlpPolicy', env, tensorboard_log='train_log', verbose=0, **model_params)
	# Train the agent
	model.learn(total_timesteps=int(1e4))

	loops = 0
	obs = env.reset()
	reward = 0 #We want the last reward to be minimal (perhaps instead do cumulative?)
	while loops < 100:
		action, states = model.predict(obs)
		(obs, reward, _, _) = env.step(action)
		loops += 1
	trial.report(-reward)
	return -reward #Optuna by default minimizes, so changing this to positive distance

if __name__ == "__main__":
	#I store the results in a SQLite database so that it can resume from checkpoints.
	study = optuna.create_study(study_name='ppo_nudging', storage='sqlite:///params.db', load_if_exists=True)
	study.optimize(optimize_algorithm, n_trials=1000)

#Training without Optuna, so that we can compare the trained model to best possible controllers
#if __name__ == "__main__":
#	graph, clusters, pos = generateGraph(3, 45, draw=False)	#Generate graph
#	#Nudging environment
#	env = gym.make('Controller-RandomPlacement-v0', graph=graph, clusters=clusters, pos=pos)
#	#act = []
#	#for i in range(3):
#	#	act.append([5 - i for i in range(env.degree)])
#	#env.render()
#	#env.step(act)
#	#Agent
#	model = PPO1('MlpPolicy', env, tensorboard_log='train_log', verbose=1)
#	# Train the agent
#	model.learn(total_timesteps=int(1e7))

#	loops = 0
#	obs = env.reset()
#	while loops < 1000:
#		action, states = model.predict(obs)
#		(obs, reward, _, _) = env.step(action)
#		print("CONTROLLERS:", env.controllers)
#		print("OBSERVATION:", obs)
#		print("REWARD: ", reward)
#		loops += 1
#	(best_controllers, best_dist) = env.calculateOptimal()
#	print("Optimal:", best_controllers)
#	print("Optimal Distance:", best_dist)
#	env.render()

#Testing best number of packets per environment step (to evaluate controller placement)
#print("Starting trials . . .")
#for i in range(10):
#	n_clusters = i + 2
#	n_nodes = (i + 2) * np.random.randint(5, 10)
#	graph, clusters, pos = generateGraph(n_clusters, n_nodes, draw=False)	#Generate graph
#	env = gym.make('Controller-v0', graph=graph, clusters=clusters, pos=pos)
#	controllers = env._random_valid_controllers()
#	num_trials = 1000
#	num_steps = 20
#	trial_data = np.zeros(shape=(num_trials, num_steps))
#	for trial in range(num_trials):
#		for step in range(num_steps):
#			trial_data[trial, step] = env.step(controllers, num_packets=10 + step * 20) / (10 + step * 20.0)
#	print(n_clusters, n_nodes)
#	print(np.mean(trial_data, axis=0))
#	print(np.std(trial_data, axis=0))