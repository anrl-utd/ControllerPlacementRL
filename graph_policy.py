# coding=utf-8
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
import tf_geometric as tfg
from tensorflow import keras
import networkx as nx
import pickle
from stable_baselines.deepq.policies import DQNPolicy
import controller_env
from controller_env.envs.graph_env import generateGraph, generateAlternateGraph, generateClusters, ControllerEnv
import random
import matplotlib.pyplot as plt
import math
import traceback
import gym
from stable_baselines import PPO1, DQN
from stable_baselines.deepq.replay_buffer import ReplayBuffer  # PrioritizedReplayBuffer vs ReplayBuffer, what is the difference?
import shutil
import sys
import signal

#graph = tfg.Graph(
#    x=np.random.randn(5, 20),  # 5 nodes, 20 features,
#    edge_index=[[0, 0, 1, 3],
#                [1, 2, 2, 1]]  # 4 undirected edges
#)

#print("Graph Desc: \n", graph)

#graph.convert_edge_to_directed()  # pre-process edges
#print("Processed Graph Desc: \n", graph)
#print("Processed Edge Index:\n", graph.edge_index)

## Multi-head Graph Attention Network (GAT)
#gat_layer = tfg.layers.GAT(units=4, num_heads=4, activation=tf.nn.relu)
#output = gat_layer([graph.x, graph.edge_index])
#print("Output of GAT: \n", output)

clusters = pickle.load(open('clusters.pickle', 'rb'))
pos = pickle.load(open('position.pickle', 'rb'))
graph = nx.read_gpickle('graph.gpickle')

tf.enable_eager_execution()

class GATPolicy(DQNPolicy):
	def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, graph):
		super(GATPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch)
		cluster_info = nx.get_node_attributes(graph, 'cluster')
		node_attr = np.zeros(shape=(len(cluster_info),2), dtype=np.float32)
		for key, value in cluster_info.items():  # Try having all 0s for not-controller
			node_attr[key][1] = value

		edge_weights = nx.get_edge_attributes(graph, 'weight')
		edge_attr = np.empty(shape=(len(edge_weights),), dtype=np.float32)
		edges = np.array(graph.edges).T
		i = 0
		for key, value in edge_weights.items():
			assert (edges[0][i], edges[1][i]) == key
			edge_attr[i] = value
			i += 1

		edges, [edge_attr] = tfg.utils.graph_utils.convert_edge_to_directed(edges, [edge_attr])
		self.graph = tfg.Graph(
			x=node_attr,
			edge_index=edges,
			edge_weight=edge_attr
		)
		print([self.graph.x, self.graph.edge_index, self.graph.edge_weight])
		print(self.graph.x.shape)
		print(self.graph.edge_index.shape)
		print(self.graph.edge_weight.shape)
		print(self.graph)
		#self.graph.x = tf.placeholder(tf.int16, shape=(len(cluster_info),))
		#print(self.graph.x)
		self.graph.convert_data_to_tensor()
		gcn0 = tfg.layers.GCN(16, activation=tf.nn.relu)
		#gcn1 = tfg.layers.GCN(128)

		h = gcn0([self.graph.x, self.graph.edge_index, self.graph.edge_weight])
		h = gcn1([h, self.graph.edge_index, self.graph.edge_weight], cache=self.graph.cache)
		print(h)
		self.q_values = action_out
		self._setup_init()

		def step(self, obs, state=None, mask=None, deterministic=True):
			q_values, actions_proba = self.sess.run([self.q_values, self.policy_proba], {self.obs_ph: obs})
			if deterministic:
				actions = np.argmax(q_values, axis=1)
			else:
				# Unefficient sampling
				# TODO: replace the loop
				# maybe with Gumbel-max trick ? (http://amid.fish/humble-gumbel)
				actions = np.zeros((len(obs),), dtype=np.int64)
				for action_idx in range(len(obs)):
					actions[action_idx] = np.random.choice(self.n_actions, p=actions_proba[action_idx])

			return actions, q_values, None

	def proba_step(self, obs, state=None, mask=None):
		return self.sess.run(self.policy_proba, {self.obs_ph: obs})

env = gym.make('Controller-Cluster-v0', graph=graph, clusters=clusters, pos={})
policy_params={'graph': graph}
model = DQN(GATPolicy, env, policy_kwargs=policy_params)
model.learn(1000)