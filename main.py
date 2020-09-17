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
from controller_env.envs.graph_env import generateGraph, generateAlternateGraph, generateClusters
import random
import matplotlib.pyplot as plt
import math
import networkx as nx
import traceback

from stable_baselines import PPO1, DQN
from stable_baselines.deepq.policies import MlpPolicy, LnMlpPolicy
from stable_baselines.deepq.replay_buffer import ReplayBuffer  # PrioritizedReplayBuffer vs ReplayBuffer, what is the difference?
import shutil
import pickle
import sys
import signal

def train_once(graph: nx.Graph, clusters: list, pos: dict, env_name: str='Controller-Select-v0', compute_optimal: bool=True, trained_model: DQN=None, steps: int=2e5, logdir: str='train_log_compare', env_kwargs: dict={}) -> DQN:
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
	print(env.reset().shape)
	env.render(mode='original_graph.png')
	optimal_controllers = None
	if compute_optimal:
		print("Computing optimal!")
		optimal_controllers = env.calculateOptimal()

	# Generate custom replay buffer full of valid experiences to speed up exploration of training
	def add_wrapper(replay_buffer):
		# Replay buffer maxsize is by default 50000. Should this be lowered?
		valid_controllers_set = [env._random_valid_controllers() for i in range(int(replay_buffer._maxsize * 0.5 / len(clusters)))]
	
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
		model = DQN(LnMlpPolicy, env, tensorboard_log=logdir, verbose=0, exploration_initial_eps=0.2, exploration_fraction=0.025, learning_starts=0, target_network_update_freq=100, batch_size=32, seed=256)
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
	while not done:
		action, states = model.predict(obs)
		(obs, rew, done, _) = env.step(action)
		reward += rew
		reward_final = rew

	# Show controllers chosen by the model
	env.render(mode='graph_end.png')
	print(env.controllers, reward_final)
	print("BEST EVER:")
	print(env.best_controllers, env.best_reward)
	best_reward = env.optimal_neighbors(graph, env.best_controllers)
	print(best_reward)

	# Show controllers chosen using heuristic
	env.reset()
	centroid_controllers, heuristic_distance = env.graphCentroidAction()
	# Convert heuristic controllers to actual
	print(centroid_controllers)
	if env_name == 'Controller-Cluster-v0' or env_name == 'Controller-Cluster-Options-v0':
		# Assume all clusters same length
		centroid_controllers.sort()
		cluster_len = len(clusters[0])
		for i in range(len(clusters)):
			centroid_controllers[i] -= i * cluster_len
	print(centroid_controllers)
	for cont in centroid_controllers:
		(_, reward_final, _, _) = env.step(cont)
	env.render(mode='graph_heuristic.png')
	best_heuristic = reward_final
	print(env.controllers, reward_final)
	print(env.optimal_neighbors(graph,  env.controllers))

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
		print("Generated {}-cluster graph!".format(len(clusters)))
		m, best_rl, best_heuristic = train_once(graph, clusters, pos, compute_optimal=False, env_name='Controller-Cluster-v0', logdir='train_log_env_compare')  # Train
		print("Best-ever RL: {}, Heuristic: {}".format(best_rl, best_heuristic))
	except Exception as e:
		print(e)
		traceback.print_exc()
		print('Interrupted, saving . . . ')
		nx.write_gpickle(graph, 'graph.gpickle')
		pickle.dump(clusters, open('clusters.pickle', 'wb'))
		pickle.dump(pos, open('position.pickle', 'wb'))