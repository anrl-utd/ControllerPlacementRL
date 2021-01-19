"""
Main code to create graph and run agent
Author: Usaid Malik

TODO: Make Optuna great again
"""
import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'  # Necessary to supress Fortran overwriting keyboard interrupt (don't ask me why, idk but it works)
import gym
import numpy as np
import controller_env
from controller_env.envs.graph_env import generateGraph, generateAlternateGraph, generateClusters, ControllerEnv
import random
import matplotlib.pyplot as plt
import math
import networkx as nx
import traceback

from stable_baselines import PPO1, DQN
from stable_baselines.deepq.policies import MlpPolicy, LnMlpPolicy, MlpLnLstmPolicy
from stable_baselines.deepq.replay_buffer import ReplayBuffer  # PrioritizedReplayBuffer vs ReplayBuffer, what is the difference?
import shutil
import pickle
import sys
import signal
import optuna

def optimize(trial, graph, clusters, heuristic_rounded, env, env_name='Controller-Cluster-v0', env_kwargs: dict={}):
	"""Optimizes algorithm uses Optuna"""
	try:
		model_params = {
			#'gamma': trial.suggest_loguniform('gamma', 0.9, 0.999),
			#'learning_rate': trial.suggest_uniform('learning_rate', 0.00005, 0.001),
			'exploration_fraction': trial.suggest_uniform('exploration_fraction', 0.05, 0.2),
			'buffer_size': trial.suggest_int('buffer_size', low=25000, high=100000, step=25000),
			'learning_starts': trial.suggest_int('learning_starts', 0, 2000, step=200),
			#'prioritized_replay': trial.suggest_categorical('prioritized_replay', [True, False]),
			#'double_q': trial.suggest_categorical('double_q', [True, False])
		}

		# Selecting controllers one-at-a-time environment
		env.reset()
		# Agent
		model = DQN(LnMlpPolicy, env, tensorboard_log='train_log_optuna', verbose=0, seed=256, double_q=True, prioritized_replay=True, learning_rate=0.001, gamma=0.9, batch_size=64, **model_params)
		model.learn(total_timesteps=int(2e5))
		#trial.report(env.best_reward)
		return env.cumulative_reward / (env.best_reward - heuristic_rounded)
	except KeyboardInterrupt:
		return Exception

def train_once(graph: nx.Graph, clusters: list, pos: dict, env_name: str='Controller-Select-v0', compute_optimal: bool=True, trained_model: DQN=None, steps: int=2e5, logdir: str='train_log_compare', env_kwargs: dict={}) -> (DQN, float, float):
	"""
	Main training loop. Initializes RL environment, performs training, and outputs results
	Args:
		graph (nx.Graph): NetworkX graph to train on
		clusters (list): List of lists of nodes in each cluster
		pos (dict): Graph rendering positions
		env_name (str): Name of Gym environment
		compute_optimal (bool): Whether to compute optimal set of controllers by brute-force
		trained_model (DQN): Provide starting model to train on
	Return:
		Trained model
	"""
	# Selecting controllers one-at-a-time environment
	env = gym.make(env_name, graph=graph, clusters=clusters, pos=pos, **env_kwargs)
	#for i in range(1000):
	#	env.reset()
	#	print(env.graph.size(weight='weight'))
	orig_graph = env.original_graph
	env.render(mode='original_graph.png')
	optimal_controllers = None
	if compute_optimal:
		print("Computing optimal!")
		optimal_controllers = env.calculateOptimal()


	# Generate custom replay buffer full of valid experiences to speed up exploration of training
	def add_wrapper(replay_buffer):
		# Replay buffer maxsize is by default 50000. Should this be lowered?
		# valid_controllers_set = [env._random_valid_controllers() for i in range(int(replay_buffer._maxsize * 0.5 / len(clusters)))]
		# Uses heuristic controller set as innitial 'random' controllers
		valid_controllers_set = env.graphCentroidAction()
	
		for valid_controllers in valid_controllers_set:
			obs_current = env.reset()  # Really strange issue - obs_current follows the change in env.state, making it equal to obs!
			for controller in valid_controllers:
				(obs, rew, done, _) = env.step(controller)
				replay_buffer.add(obs_current, controller, rew, obs, done)  # For some reason, obs is a pointer which ends up being the very last obs before reset, so need to copy
				obs_current = obs.copy()
		return replay_buffer

	# Agent
	model = None
	if trained_model is None:
		print("Creating new training model!")
		model = DQN(LnMlpPolicy, env, tensorboard_log=logdir, verbose=0, exploration_initial_eps=0.2, exploration_fraction=0.025, learning_starts=0, target_network_update_freq=100, batch_size=32, learning_rate=0.00025, seed=256)
	else:
		print("Using provided training model!")
		model = trained_model
		model.set_env(env)
		model.tensorboard_log = logdir

	# Train the agent
	print("Training!")
	model.learn(total_timesteps=int(steps))#, replay_wrapper=add_wrapper)

	# Run a single run to evaluate the DQN
	obs = env.reset()
	reward = 0 #We want the last reward to be minimal (perhaps instead do cumulative?)
	reward_final = 0
	done = False
	action = None
	final_rl_actions = []
	while not done:
		action, states = model.predict(obs)
		(obs, rew, done, _) = env.step(action)
		final_rl_actions.append(action)
		reward += rew
		reward_final = rew

	# Show controllers chosen by the model
	env.render(mode='graph_end.png')
	print(env.controllers, reward_final)
	print("BEST EVER:")
	print(env.best_controllers, env.best_reward)
	best_reward = env.optimal_neighbors(graph, env.best_controllers)
	print(best_reward)

	print("HEURISTIC ON AVERAGE CHANGE:")
	for edge in orig_graph.edges:
		env.graph.edges[edge]['weight'] = orig_graph.edges[edge]['weight'] + (env.graph.edges[edge]['weight'] - orig_graph.edges[edge]['weight']) / env.num_resets
	rl_controllers = env.best_controllers
	if env_name == 'Controller-Cluster-v0':
		rl_controllers.sort()
		cluster_len = len(clusters[0])
		for i in range(len(clusters)):
			rl_controllers[i] -= i * cluster_len
	env.reset(adjust=False)
	for cont in rl_controllers:
		(_, reward_final, _, _) = env.step(cont)
	print("RL on average change graph {} - {}".format(env.best_controllers, reward_final))
	# Show controllers chosen using heuristic
	centroid_controllers, heuristic_distance = env.graphCentroidAction()
	# Convert heuristic controllers to actual
	if env_name == 'Controller-Cluster-v0' or env_name == 'Controller-Cluster-Options-v0':
		# Assume all clusters same length
		centroid_controllers.sort()
		cluster_len = len(clusters[0])
		for i in range(len(clusters)):
			centroid_controllers[i] -= i * cluster_len
	env.reset(adjust=False)
	for cont in centroid_controllers:
		(_, reward_final, _, _) = env.step(cont)
	env.render(mode='graph_heuristic.png')
	best_heuristic = reward_final
	print("Heuristic on average change graph {} - {}".format(env.controllers, reward_final))
	#print("Heuristic optimal {} - {}".format(*env.optimal_neighbors(graph,  env.controllers)))
	heuristic_controllers = env.controllers

	rl_rewards = []
	heuristic_rewards = []
	NUM_GRAPHS = 100
	for i in range(NUM_GRAPHS):
		rl_reward = None
		heuristic_reward = None
		env.reset()
		for cont in final_rl_actions:
			(_, rl_reward, _, _) = env.step(cont)
		env.reset(adjust=False)
		for cont in centroid_controllers:
			(_, heuristic_reward, _, _) = env.step(cont)
		print("RL REWARD, HEURISTIC: {}\t{}".format(rl_reward, heuristic_reward))
		rl_rewards.append(rl_reward)
		heuristic_rewards.append(heuristic_reward)

	def create_hist(fig, data, title=None, color=None):
		bins = np.arange(0, 2000, 100)
		plt.xlim([min(data) - 100, max(data) + 100])
		fig.hist(data, bins=bins, alpha=0.5, color=color)
		if title:
			fig.title(title)
		plt.xlabel('Controller Distances')
		plt.ylabel('Count')
	fig = plt.figure()
	ax1 = fig.add_subplot(2, 1, 1)
	create_hist(ax1, rl_rewards, color='blue')
	create_hist(ax1, heuristic_rewards, color='red')
	ax2 = fig.add_subplot(2, 1, 2)
	ax2.plot(np.arange(0, NUM_GRAPHS, 1), rl_rewards, c='blue')
	ax2.plot(np.arange(0, NUM_GRAPHS, 1), heuristic_rewards, c='red')
	plt.show()
	# Show optimal
	if optimal_controllers is not None:
		env.reset()
		for cont in optimal_controllers[0]:
			(_, reward_final, _, _) = env.step(cont)
		env.render(mode='graph_optimal.png')
		print(env.controllers, reward_final)
		print(optimal_controllers)
	return model, best_reward, best_heuristic

if __name__ == "__main__":
	graph = None
	clusters = None
	pos = None
	np.random.seed(256)
	# This might be lazy code, but I think it is not worth importing more modules just to check if file exists before trying to open it
	if os.path.isfile('clusters.pickle') and os.path.isfile('graph.gpickle') and os.path.isfile('position.pickle'):
		print("Found graph from file, using saved graph")
		clusters = pickle.load(open('clusters.pickle', 'rb'))
		pos = pickle.load(open('position.pickle', 'rb'))
		graph = nx.read_gpickle('graph.gpickle')
		print(clusters)
	else:
		print("Generating graph")
		graph, clusters, pos = generateAlternateGraph(6, 180)
		nx.write_gpickle(graph, 'graph.gpickle')
		pickle.dump(clusters, open('clusters.pickle', 'wb'))
		pickle.dump(pos, open('position.pickle', 'wb'))

	try:
		# Get TopologyZoo graph to train on, using edge label LinkSpeed for edge weight
		#k_graph = nx.graphml.read_graphml('Uninett2010.graphml')
		# Randomly-generate clusters
		#graph, clusters, pos = generateClusters(k_graph, edge_label='LinkSpeed')
		#print("Generated {}-cluster graph!".format(len(clusters)))
		#m, best_rl, best_heuristic = train_once(graph, clusters, pos, compute_optimal=False, env_name='Controller-Cluster-v0', logdir='train_log_env_compare')  # Train
		#print("Best-ever RL: {}, Heuristic: {}".format(best_rl, best_heuristic))
		# Get heuristic value so you don't have to repeat for every train

		# Optuna stuff
		#heuristic_env = ControllerEnv(graph, clusters)
		#heuristic_controllers, heuristic = heuristic_env.graphCentroidAction()
		#print("Heuristic: ")
		#print(heuristic_controllers, heuristic)
		#study = optuna.create_study(study_name='dqn_optimize', storage='sqlite:///params_dqn.db', load_if_exists=True)
		#env = gym.make('Controller-Cluster-v0', graph=graph, clusters=clusters, pos={})
		#study.optimize(lambda trial: optimize(trial, graph, clusters, heuristic - (heuristic % 100), env), show_progress_bar=True)
		#print("STUDY BEST PARAMS: ", study.best_params)
		#print("STUDY BEST EVER: ", study.best_value)
		#print("STUDY BEST TRIAL: ", study.best_trial)
		#print("NUM TRIALS: ", len(study.trials))

		train_once(graph, clusters, pos, 'Controller-Cluster-v0', False, steps=5e5)

	except Exception as e:
		print(e)
		traceback.print_exc()
		print('Interrupted, saving . . . ')
		nx.write_gpickle(graph, 'graph.gpickle')
		pickle.dump(clusters, open('clusters.pickle', 'wb'))
		pickle.dump(pos, open('position.pickle', 'wb'))