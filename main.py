"""
Main code to create graph and run agent
"""
import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'  # Necessary to supress Fortran overwriting keyboard interrupt (don't ask me why, idk but it works)
import gym
import numpy as np
import controller_env
from controller_env.envs.graph_env import generateGraph
import random
import matplotlib.pyplot as plt
import math
import networkx as nx

from stable_baselines import PPO1, DQN
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines.deepq.replay_buffer import ReplayBuffer  # PrioritizedReplayBuffer vs ReplayBuffer, what is the difference?
import optuna
import shutil
import pickle
import sys
import signal

def optimize_algorithm(trial, graph, clusters, pos, env_name='Controller-Select-v0'):
	"""Optimizes an algorithm using Optuna (tries out different parameters)"""
	#TODO: Ensure early pruning of trials occurs to speed up optimization (Tensorflow hook?)
	try:
		model_params = {
			#'gamma': trial.suggest_loguniform('gamma', 0.9, 0.999),
			'entcoeff': trial.suggest_loguniform('entcoeff', 0.01, 0.1),
			'lam': trial.suggest_uniform('lam', 0.9, 1),
			'clip_param': trial.suggest_uniform('clip', 0.1, 0.4)
		}

		#Nudging environment
		env = gym.make(env_name, graph=graph, clusters=clusters, pos=pos)
		#Agent
		model = PPO1('MlpPolicy', env, tensorboard_log='train_log', verbose=0, **model_params)
		# Train the agent
		model.learn(total_timesteps=int(1e4))

		loops = 0
		obs = env.reset()
		reward = 0 #We want the last reward to be minimal (perhaps instead do cumulative?)
		done = False
		while not done:
			action, states = model.predict(obs)
			(obs, rew, done, _) = env.step(action)
			reward += rew
			loops += 1
		print(np.argwhere(obs))
		trial.report(-reward)

		if(reward >= 100000):
			#Since I am unsure if Optuna does multiprocessing, I'm going to search the logging path for the last-logged
			#and delete it if it isn't relevant
			path = os.path.abspath(os.getcwd()) + '/train_log'
			list_subfolders_with_paths = [f.path for f in os.scandir(path) if f.is_dir()]
			paths = [int(f.split('/')[-1].split('_')[-1]) for f in list_subfolders_with_paths] #Can combine with previous line, but wanted readability
			try:
				shutil.rmtree(path + '/PPO1_' + str(sorted(paths)[-1]))
			except OSError as e:
				print("Error removing log file")
		return -reward #Optuna by default minimizes, so changing this to positive distance
	except KeyboardInterrupt: #Bug in Python multiprocessing, only Exceptions go through to main process
		raise Exception

def train_once(graph, clusters, pos, env_name='Controller-Select-v0', compute_optimal=True):
	#Nudging environment
	env = gym.make(env_name, graph=graph, clusters=clusters, pos=pos)
	env.reset()
	env.render()
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
		print(replay_buffer._storage[:7])
		return replay_buffer

	#Agent
	model = DQN(MlpPolicy, env, tensorboard_log='train_log_watts', verbose=0, exploration_initial_eps=0.2, exploration_fraction=0.025, learning_starts=0, target_network_update_freq=100, batch_size=32)
	# Train the agent
	print("Training!")
	model.learn(total_timesteps=int(1e6), replay_wrapper=add_wrapper)

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
	print(env.optimal_neighbors(graph, env.best_controllers))

	# Show controllers chosen using heuristic
	env.reset()
	centroid_controllers = env.graphCentroidAction()
	for cont in centroid_controllers:
		(_, reward_final, _, _) = env.step(cont)
	env.render(mode='graph_heuristic.png')
	print(env.controllers, reward_final)
	print(env.optimal_neighbors(graph,  env.controllers))

	#Show optimal
	if optimal_controllers is not None:
		env.reset()
		for cont in optimal_controllers[0]:
			(_, reward_final, _, _) = env.step(cont)
		env.render(mode='graph_optimal.png')
		print(env.controllers, reward_final)
		print(optimal_controllers)

if __name__ == "__main__":
	graph = None
	clusters = None
	pos = None
	#This might be lazy code, but I think it is not worth importing more modules just to check if file exists before trying to open it
	if os.path.isfile('clusters.pickle') and os.path.isfile('graph.gpickle') and os.path.isfile('position.pickle'):
		print("Found graph from file, using saved graph")
		clusters = pickle.load(open('clusters.pickle', 'rb'))
		pos = pickle.load(open('position.pickle', 'rb'))
		graph = nx.read_gpickle('graph.gpickle')
	else:
		print("Generating graph")
		graph, clusters, pos = generateGraph(6, 90, draw=False)
		nx.write_gpickle(graph, 'graph.gpickle')
		pickle.dump(clusters, open('clusters.pickle', 'wb'))
		pickle.dump(pos, open('position.pickle', 'wb'))
	
	try:
		#I store the results in a SQLite database so that it can resume from checkpoints.
		#study = optuna.create_study(study_name='ppo_direct', storage='sqlite:///params_select.db', load_if_exists=True)
		#study.optimize(lambda trial: optimize_algorithm(trial, graph, clusters, pos), n_trials=500)
		train_once(graph, clusters, pos, compute_optimal=False)
	except Exception as e:
		print(e)
		print('Interrupted, saving . . . ')
		nx.write_gpickle(graph, 'graph.gpickle')
		pickle.dump(clusters, open('clusters.pickle', 'wb'))
		pickle.dump(pos, open('position.pickle', 'wb'))
	

#DQN trials: DQN_15 for DQN_3C_45N_Replay_1, DQN_18 for DQN_3C_45N_NoReplay_1

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