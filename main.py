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
import networkx as nx

from stable_baselines import PPO1
import optuna
import shutil
import pickle
import os

def optimize_algorithm(trial, graph, clusters, env_name='Controller-Direct-v0'):
	"""Optimizes an algorithm using Optuna (tries out different parameters)"""
	#TODO: Ensure early pruning of trials occurs to speed up optimization (Tensorflow hook?)

	model_params = {
		'gamma': trial.suggest_loguniform('gamma', 0.9, 0.999),
		'adam_epsilon': trial.suggest_uniform('adam_epsilon', 1e-8, 1e-4),
		'lam': trial.suggest_uniform('lam', 0.9, 1),
		'clip_param': trial.suggest_uniform('clip', 0.1, 0.4)
	}

	#Nudging environment
	env = gym.make('Controller-Direct-v0', graph=graph, clusters=clusters, pos=pos)
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
	trial.report(reward)

	if(reward == 100000):
		#Since I am unsure if Optuna does multiprocessing, I'm going to search the logging path for the last-logged
		#and delete it if it isn't relevant
		path = os.path.abspath(os.getcwd()) + '/train_log'
		list_subfolders_with_paths = [f.path for f in os.scandir(path) if f.is_dir()]
		paths = [int(f.split('/')[-1].split('_')[-1]) for f in list_subfolders_with_paths] #Can combine with previous line, but wanted readability
		try:
			shutil.rmtree(path + '/PPO1_' + str(sorted(paths)[-1]))
		except OSError as e:
			print("Error removing log file")
	return reward #Optuna by default minimizes, so changing this to positive distance

if __name__ == "__main__":
	graph = None
	clusters = None
	#This might be lazy code, but I think it is not worth importing more modules just to check if file exists before trying to open it
	if os.path.isfile('clusters.pickle') and os.path.isfile('graph.gpickle'):
		clusters = pickle.load(open('clusters.pickle', 'rb'))
		graph = nx.read_gpickle('graph.gpickle')
	else:
		graph, clusters, pos = generateGraph(3, 45, draw=False)
	try:
		#I store the results in a SQLite database so that it can resume from checkpoints.
		study = optuna.create_study(study_name='ppo_direct', storage='sqlite:///params_direct.db', load_if_exists=True)
		study.optimize(lambda trial: optimize_algorithm(trial, graph, clusters), n_trials=500)
	except KeyboardInterrupt:
		print('Interrupted, saving . . . ')
		nx.write_gpickle(graph, 'graph.gpickle')
		pickle.dump(clusters, open('clusters.pickle', 'wb'))
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