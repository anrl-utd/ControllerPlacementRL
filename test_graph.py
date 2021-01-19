from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import time
import sys

from absl import app
from absl import flags
from absl import logging

import gin
from six.moves import range
import tensorflow as tf	# pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.agents.dqn import dqn_agent
from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.agents import DdpgAgent
import tf_agents.agents.ddpg as ddpg
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.environments.examples import masked_cartpole	# pylint: disable=unused-import
from tf_agents.eval import metric_utils
from tf_agents.keras_layers import dynamic_unroll_layer
from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential, network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
from tf_agents.networks import actor_distribution_network
import controller_env
import pickle
import networkx as nx
import stellargraph as sg
from stellargraph.mapper import FullBatchNodeGenerator
from stellargraph.layer import GAT
import pandas as pd
import numpy as np
from spektral.layers import GCNConv, GATConv
from spektral.utils import gcn_filter
from spektral.layers.ops import dot, filter_dot
from spektral.layers.ops.modes import autodetect_mode

from tensorflow.keras import backend as K

from spektral.layers import ops
from spektral.layers.convolutional.conv import Conv
from spektral.utils import gcn_filter


#class GCNConv(Conv):
#	r"""
#	A graph convolutional layer (GCN) from the paper
#	> [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)<br>
#	> Thomas N. Kipf and Max Welling
#	**Mode**: single, disjoint, mixed, batch.
#	This layer computes:
#	$$
#		\X' = \hat \D^{-1/2} \hat \A \hat \D^{-1/2} \X \W + \b
#	$$
#	where \( \hat \A = \A + \I \) is the adjacency matrix with added self-loops
#	and \(\hat\D\) is its degree matrix.
#	**Input**
#	- Node features of shape `([batch], n_nodes, n_node_features)`;
#	- Modified Laplacian of shape `([batch], n_nodes, n_nodes)`; can be computed with
#	`spektral.utils.convolution.gcn_filter`.
#	**Output**
#	- Node features with the same shape as the input, but with the last
#	dimension changed to `channels`.
#	**Arguments**
#	- `channels`: number of output channels;
#	- `activation`: activation function;
#	- `use_bias`: bool, add a bias vector to the output;
#	- `kernel_initializer`: initializer for the weights;
#	- `bias_initializer`: initializer for the bias vector;
#	- `kernel_regularizer`: regularization applied to the weights;
#	- `bias_regularizer`: regularization applied to the bias vector;
#	- `activity_regularizer`: regularization applied to the output;
#	- `kernel_constraint`: constraint applied to the weights;
#	- `bias_constraint`: constraint applied to the bias vector.
#	"""

#	def __init__(self,
#				 channels,
#				 activation=None,
#				 use_bias=True,
#				 kernel_initializer='glorot_uniform',
#				 bias_initializer='zeros',
#				 kernel_regularizer=None,
#				 bias_regularizer=None,
#				 activity_regularizer=None,
#				 kernel_constraint=None,
#				 bias_constraint=None,
#				 **kwargs):
#		super().__init__(activation=activation,
#						 use_bias=use_bias,
#						 kernel_initializer=kernel_initializer,
#						 bias_initializer=bias_initializer,
#						 kernel_regularizer=kernel_regularizer,
#						 bias_regularizer=bias_regularizer,
#						 activity_regularizer=activity_regularizer,
#						 kernel_constraint=kernel_constraint,
#						 bias_constraint=bias_constraint,
#						 **kwargs)
#		self.channels = channels

#	def build(self, input_shape):
#		assert len(input_shape) >= 2
#		input_dim = input_shape[0][-1]
#		self.kernel = self.add_weight(shape=(input_dim, self.channels),
#									  initializer=self.kernel_initializer,
#									  name='kernel',
#									  regularizer=self.kernel_regularizer,
#									  constraint=self.kernel_constraint)
#		if self.use_bias:
#			self.bias = self.add_weight(shape=(self.channels,),
#										initializer=self.bias_initializer,
#										name='bias',
#										regularizer=self.bias_regularizer,
#										constraint=self.bias_constraint)
#		else:
#			self.bias = None
#		self.built = True

#	def call(self, inputs):
#		x, a = inputs
#		tf.assert_rank(x, 3)
#		output = ops.dot(x, self.kernel)
#		tf.print(tf.shape(output), output_stream=sys.stdout)
#		tf.print(tf.shape(a), output_stream=sys.stdout)
#		tf.print(tf.rank(a), output_stream=sys.stdout)
#		tf.print(tf.rank(output), output_stream=sys.stdout)
#		tf.assert_rank(output, 3)
#		output = tf.matmul(a, output, transpose_a=False, transpose_b=False)

#		if self.use_bias:
#			output = K.bias_add(output, self.bias)
#		output = self.activation(output)

#		return output

#	@property
#	def config(self):
#		return {
#			'channels': self.channels
#		}

#	@staticmethod
#	def preprocess(a):
#		return gcn_filter(a)

flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
					'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_integer('num_iterations', 500000,
					 'Total number train/eval iterations to perform.')
flags.DEFINE_multi_string('gin_file', None, 'Paths to the gin-config files.')
flags.DEFINE_multi_string('gin_param', None, 'Gin binding parameters.')

FLAGS = flags.FLAGS

KERAS_LSTM_FUSED = 2

@gin.configurable
def train_eval(
	root_dir,
	env_name='CartPole-v0',
	num_iterations=5e5,
	train_sequence_length=1,
	# Params for QNetwork
	fc_layer_params=(64,64,),
	# Params for QRnnNetwork
	input_fc_layer_params=(50,),
	lstm_size=(6,),
	output_fc_layer_params=(30,),

	# Params for collect
	initial_collect_steps=2000,
	collect_steps_per_iteration=6,
	epsilon_greedy=0.1,
	replay_buffer_capacity=100000,
	# Params for target update
	target_update_tau=0.05,
	target_update_period=5,
	# Params for train
	train_steps_per_iteration=6,
	batch_size=32,
	learning_rate=1e-3,
	n_step_update=1,
	gamma=0.99,
	reward_scale_factor=1.0,
	gradient_clipping=None,
	use_tf_functions=True,
	# Params for eval
	num_eval_episodes=1,
	eval_interval=1000,
	# Params for checkpoints
	train_checkpoint_interval=10000,
	policy_checkpoint_interval=5000,
	rb_checkpoint_interval=20000,
	# Params for summaries and logging
	log_interval=1000,
	summary_interval=1000,
	summaries_flush_secs=10,
	debug_summaries=False,
	summarize_grads_and_vars=False,
	eval_metrics_callback=None):
	"""A simple train and eval for DQN."""
	root_dir = os.path.expanduser(root_dir)
	train_dir = os.path.join(root_dir, 'train')
	eval_dir = os.path.join(root_dir, 'eval')
	clusters = pickle.load(open('clusters.pickle', 'rb'))
	graph = nx.read_gpickle('graph.gpickle')
	print(graph.nodes)
	train_summary_writer = tf.compat.v2.summary.create_file_writer(
		train_dir, flush_millis=summaries_flush_secs * 1000)
	train_summary_writer.set_as_default()

	eval_summary_writer = tf.compat.v2.summary.create_file_writer(
		eval_dir, flush_millis=summaries_flush_secs * 1000)
	eval_metrics = [
		tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
		tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes)
	]

	global_step = tf.compat.v1.train.get_or_create_global_step()
	with tf.compat.v2.summary.record_if(
		lambda: tf.math.equal(global_step % summary_interval, 0)):
		tf_env = tf_py_environment.TFPyEnvironment(suite_gym.load(env_name, gym_kwargs={
		'graph': graph,
		'clusters': clusters}))
		eval_tf_env = tf_py_environment.TFPyEnvironment(suite_gym.load(env_name, gym_kwargs={
		'graph': graph,
		'clusters': clusters}))

		if train_sequence_length != 1 and n_step_update != 1:
			raise NotImplementedError(
				'train_eval does not currently support n-step updates with stateful '
				'networks (i.e., RNNs)')

		action_spec = tf_env.action_spec()
		num_actions = action_spec.maximum - action_spec.minimum + 1

		if train_sequence_length > 1:
			q_net = create_recurrent_network(
				input_fc_layer_params,
				lstm_size,
				output_fc_layer_params,
				num_actions)
		else:
			q_net = create_feedforward_network(fc_layer_params, num_actions)
			train_sequence_length = n_step_update
		q_net = GATNetwork(tf_env.observation_spec(), tf_env.action_spec(), graph)
		#time_step = tf_env.reset()
		#q_net(time_step.observation, time_step.step_type)
		#q_net = actor_distribution_network.ActorDistributionNetwork(
		#	tf_env.observation_spec(),
		#	tf_env.action_spec(),
		#	fc_layer_params=fc_layer_params)

		#q_net = QNetwork(tf_env.observation_spec(), tf_env.action_spec(), 30)
		# TODO(b/127301657): Decay epsilon based on global step, cf. cl/188907839
		tf_agent = dqn_agent.DqnAgent(
			tf_env.time_step_spec(),
			tf_env.action_spec(),
			q_network=q_net,
			epsilon_greedy=epsilon_greedy,
			n_step_update=n_step_update,
			target_update_tau=target_update_tau,
			target_update_period=target_update_period,
			optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate),
			td_errors_loss_fn=common.element_wise_squared_loss,
			gamma=gamma,
			reward_scale_factor=reward_scale_factor,
			gradient_clipping=gradient_clipping,
			debug_summaries=debug_summaries,
			summarize_grads_and_vars=summarize_grads_and_vars,
			train_step_counter=global_step)
		#critic_net = ddpg.critic_network.CriticNetwork(
		#(tf_env.observation_spec(), tf_env.action_spec()),
		#observation_fc_layer_params=None,
		#action_fc_layer_params=None,
		#joint_fc_layer_params=(64,64,),
		#kernel_initializer='glorot_uniform',
		#last_kernel_initializer='glorot_uniform')

		#tf_agent = DdpgAgent(tf_env.time_step_spec(),
		#			   tf_env.action_spec(),
		#			   actor_network=q_net,
		#			   critic_network=critic_net,
		#			   actor_optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate),
		#			   critic_optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate),
		#			   ou_stddev=0.0,
		#			   ou_damping=0.0)
		tf_agent.initialize()

		train_metrics = [
			tf_metrics.NumberOfEpisodes(),
			tf_metrics.EnvironmentSteps(),
			tf_metrics.AverageReturnMetric(),
			tf_metrics.AverageEpisodeLengthMetric(),
			tf_metrics.MaxReturnMetric(),
		]

		eval_policy = tf_agent.policy
		collect_policy = tf_agent.collect_policy

		replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
			data_spec=tf_agent.collect_data_spec,
			batch_size=tf_env.batch_size,
			max_length=replay_buffer_capacity)

		collect_driver = dynamic_step_driver.DynamicStepDriver(
			tf_env,
			collect_policy,
			observers=[replay_buffer.add_batch] + train_metrics,
			num_steps=collect_steps_per_iteration)

		train_checkpointer = common.Checkpointer(
			ckpt_dir=train_dir,
			agent=tf_agent,
			global_step=global_step,
			metrics=metric_utils.MetricsGroup(train_metrics, 'train_metrics'))
		policy_checkpointer = common.Checkpointer(
			ckpt_dir=os.path.join(train_dir, 'policy'),
			policy=eval_policy,
			global_step=global_step)
		rb_checkpointer = common.Checkpointer(
			ckpt_dir=os.path.join(train_dir, 'replay_buffer'),
			max_to_keep=1,
			replay_buffer=replay_buffer)

		train_checkpointer.initialize_or_restore()
		rb_checkpointer.initialize_or_restore()

		if use_tf_functions:
			# To speed up collect use common.function.
			collect_driver.run = common.function(collect_driver.run)
			tf_agent.train = common.function(tf_agent.train)

		initial_collect_policy = random_tf_policy.RandomTFPolicy(
			tf_env.time_step_spec(), tf_env.action_spec())

		# Collect initial replay data.
		logging.info(
			'Initializing replay buffer by collecting experience for %d steps with '
			'a random policy.', initial_collect_steps)
		dynamic_step_driver.DynamicStepDriver(
			tf_env,
			initial_collect_policy,
			observers=[replay_buffer.add_batch] + train_metrics,
			num_steps=initial_collect_steps).run()

		results = metric_utils.eager_compute(
			eval_metrics,
			eval_tf_env,
			eval_policy,
			num_episodes=num_eval_episodes,
			train_step=global_step,
			summary_writer=eval_summary_writer,
			summary_prefix='Metrics',
		)
		if eval_metrics_callback is not None:
			eval_metrics_callback(results, global_step.numpy())
		metric_utils.log_metrics(eval_metrics)

		time_step = None
		policy_state = collect_policy.get_initial_state(tf_env.batch_size)

		timed_at_step = global_step.numpy()
		time_acc = 0

		# Dataset generates trajectories with shape [Bx2x...]
		dataset = replay_buffer.as_dataset(
			num_parallel_calls=3,
			sample_batch_size=batch_size,
			num_steps=train_sequence_length + 1).prefetch(3)
		iterator = iter(dataset)

		def train_step():
			experience, _ = next(iterator)
			return tf_agent.train(experience)

		if use_tf_functions:
			train_step = common.function(train_step)

		for _ in range(num_iterations):
			start_time = time.time()
			time_step, policy_state = collect_driver.run(
				time_step=time_step,
				policy_state=policy_state,
			)
			for _ in range(train_steps_per_iteration):
				train_loss = train_step()
			time_acc += time.time() - start_time

			if global_step.numpy() % log_interval == 0:
				logging.info('step = %d, loss = %f', global_step.numpy(),
							 train_loss.loss)
				steps_per_sec = (global_step.numpy() - timed_at_step) / time_acc
				logging.info('%.3f steps/sec', steps_per_sec)
				tf.compat.v2.summary.scalar(
					name='global_steps_per_sec', data=steps_per_sec, step=global_step)
				timed_at_step = global_step.numpy()
				time_acc = 0

			for train_metric in train_metrics:
				train_metric.tf_summaries(
					train_step=global_step, step_metrics=train_metrics[:2])

			if global_step.numpy() % train_checkpoint_interval == 0:
				train_checkpointer.save(global_step=global_step.numpy())

			if global_step.numpy() % policy_checkpoint_interval == 0:
				policy_checkpointer.save(global_step=global_step.numpy())

			if global_step.numpy() % rb_checkpoint_interval == 0:
				rb_checkpointer.save(global_step=global_step.numpy())

			if global_step.numpy() % eval_interval == 0:
				results = metric_utils.eager_compute(
					eval_metrics,
					eval_tf_env,
					eval_policy,
					num_episodes=num_eval_episodes,
					train_step=global_step,
					summary_writer=eval_summary_writer,
					summary_prefix='Metrics',
				)
				if eval_metrics_callback is not None:
					eval_metrics_callback(results, global_step.numpy())
				metric_utils.log_metrics(eval_metrics)
		print(tf_env.envs[0]._gym_env.best_controllers)
		print(tf_env.envs[0]._gym_env.best_reward)
		tf_env.envs[0]._gym_env.reset()
		centroid_controllers, heuristic_distance = tf_env.envs[0]._gym_env.graphCentroidAction()
		# Convert heuristic controllers to actual
		print(centroid_controllers)
		# Assume all clusters same length
		#centroid_controllers.sort()
		#cluster_len = len(clusters[0])
		#for i in range(len(clusters)):
		#	centroid_controllers[i] -= i * cluster_len
		print(centroid_controllers)
		for cont in centroid_controllers:
			(_, reward_final, _, _) = tf_env.envs[0]._gym_env.step(cont)
		best_heuristic = reward_final
		print(tf_env.envs[0]._gym_env.controllers, reward_final)
		return train_loss


logits = functools.partial(
	tf.keras.layers.Dense,
	activation=None,
	kernel_initializer=tf.compat.v1.initializers.random_uniform(
		minval=-0.03, maxval=0.03),
	bias_initializer=tf.compat.v1.initializers.constant(-0.2))


dense = functools.partial(
	tf.keras.layers.Dense,
	activation=tf.keras.activations.relu,
	kernel_initializer=tf.compat.v1.variance_scaling_initializer(
		scale=2.0, mode='fan_in', distribution='truncated_normal'))

fused_lstm_cell = functools.partial(
	tf.keras.layers.LSTMCell, implementation=KERAS_LSTM_FUSED)

class GATNetwork(network.Network):
	def __init__(self, input_tensor_spec, action_spec, graph, num_actions=30, name=None):
		super(GATNetwork, self).__init__(
		input_tensor_spec=input_tensor_spec,
		state_spec=(),
		name=name)
		#features = pd.DataFrame(
		#	{'cluster': [i // 30 for i in graph.nodes], 'controller': [0 for i in graph.nodes]}, index=graph.nodes)
		#sg_graph = sg.StellarGraph.from_networkx(graph, node_features=features)
		
		#print(sg_graph.info())
		#generator = FullBatchNodeGenerator(sg_graph, 'gat')
		#print(generator.Aadj)
		#self.tf_clusters = tf.constant(generator.features[:, 0][np.newaxis, ...], dtype=tf.int32)
		#self.net = GAT(
		#	layer_sizes=[64, 30],
		#	activations=['relu', 'softmax'],
		#	attn_heads=8,
		#	generator=generator,
		#	in_dropout=0.5,
		#	attn_dropout=0.5,
		#	normalize=None
		#)
		#self.inp, self.out = self.net.in_out_tensors()
		self.graph_conv_1 = GCNConv(64)
		#self.graph_conv_2 = GCNConv(64)
		self.graph_conv_3 = GCNConv(1)
		#self.graph_conv_1.build(((1, 180, 1), (1, 180, 180)))
		#self.dropout_1 = tf.keras.layers.Dropout(0.5)
		self.dense_1 = dense(180)
		#self.graph_gat_1 = GATConv(16, 6)
		#self.graph_gat_2 = GATConv(1)
		adj_matrix = nx.to_numpy_array(graph)
		idn_matrix = np.identity(adj_matrix.shape[0])
		adj_loop_matrix = adj_matrix + idn_matrix
		self.adjacency_matrix = adj_loop_matrix[np.newaxis, ...]
		self.gcn_adj = gcn_filter(self.adjacency_matrix)
		self.tf_adj = tf.constant(self.adjacency_matrix)
		self.tf_gcn_adj = tf.constant(self.gcn_adj)
		self.flatten = tf.keras.layers.Flatten()

	def call(self, inputs, step_type=None, network_state=()):
		del step_type
		#stacked = []
		#for layer in tf.unstack(inputs):
		#	exp_layer = tf.expand_dims(layer, axis=0)
		#	stacked.append(tf.stack([self.tf_clusters, exp_layer], axis=-1))
		#inputs = tf.concat(stacked, axis=0)

		#inputs = tf.stack([self.tf_clusters, inputs], axis=-1)  # Single, not batch

		inputs = tf.cast(inputs, tf.float32)
		inputs = tf.expand_dims(inputs, axis=-1)
		tf_adj_repeated = tf.repeat(self.tf_gcn_adj, tf.shape(inputs)[0], axis=0)
		#tf.print('INPUTS: {}'.format(tf.shape(inputs)), output_stream=sys.stdout)
		#tf.print('ADJ: {}'.format(tf.shape(tf_adj)), output_stream=sys.stdout)
		#print(tf.reduce_sum(inputs).numpy())
		#features = tf.cast(inputs, tf.float32)
		#action_out = tf.keras.layers.Flatten()(features)
		#out = self.graph_conv_1([inputs, tf_adj])
		#output = dot(inputs, self.graph_conv_1.kernel)
		#print(tf.keras.backend.ndim(output))
		#print(autodetect_mode(tf_adj, output))
		#output = tf.cast(output, tf.double)
		#out = dot(tf_adj, output)
		#out = self.dropout_1(out)
		out = self.graph_conv_3([inputs, tf_adj_repeated])
		#out = self.graph_gat_2([out, tf_adj])
		#out = self.graph_gat_1([inputs, self.tf_adj])
		#tf.print("CONV {}".format(out), output_stream=sys.stdout)
		#out = self.graph_gat_2([inputs, self.tf_adj])
		#print(out)
		#out = tf.keras.layers.Flatten()(out)
		#q_out = self.dense_1(out)
		q_out = self.flatten(out)
		#print(tf.shape(q_out))
		#q_out = tf.math.argmax(q_out, axis=1)
		return q_out, network_state

class QNetwork(network.Network):
	def __init__(self, input_tensor_spec, action_spec, num_actions=6, name=None):
		super(QNetwork, self).__init__(
		input_tensor_spec=input_tensor_spec,
		state_spec=(),
		name=name)
		self._sub_layers = [
			tf.keras.layers.Dense(64),
			tf.keras.layers.Dense(64),
		]

	def call(self, inputs, step_type=None, network_state=()):
		del step_type
		print('Call')
		print(inputs)
		features = tf.cast(inputs, tf.float32)
		action_out = tf.keras.layers.Flatten()(features)
		for layer in self._sub_layers:
			action_out = layer(action_out)
			action_out = tf.keras.layers.LayerNormalization()(action_out)
			action_out = tf.nn.relu(action_out)
		action_out = tf.keras.layers.Dense(30)(action_out)
		state_out = features
		for layer in self._sub_layers:
			state_out = layer(state_out)
			state_out = tf.keras.layers.LayerNormalization()(state_out)
			state_out = tf.nn.relu(state_out)
		state_out = tf.keras.layers.Dense(30)(state_out)
		action_scores_mean = tf.reduce_mean(action_out, axis=1)
		action_scores_centered = action_out - tf.expand_dims(action_scores_mean, axis=1)
		q_out = state_out + action_scores_centered
		return q_out, network_state

def create_feedforward_network(fc_layer_units, num_actions):
	return sequential.Sequential(
		[dense(num_units) for num_units in fc_layer_units]
		+ [logits(num_actions)])


def create_recurrent_network(
	input_fc_layer_units,
	lstm_size,
	output_fc_layer_units,
	num_actions):
	rnn_cell = tf.keras.layers.StackedRNNCells(
		[fused_lstm_cell(s) for s in lstm_size])
	return sequential.Sequential(
		[dense(num_units) for num_units in input_fc_layer_units]
		+ [dynamic_unroll_layer.DynamicUnroll(rnn_cell)]
		+ [dense(num_units) for num_units in output_fc_layer_units]
		+ [logits(num_actions)])

def main(_):
	logging.set_verbosity(logging.INFO)
	tf.compat.v1.enable_v2_behavior()
	gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
	train_eval(FLAGS.root_dir, 'Controller-Single-v0', num_iterations=FLAGS.num_iterations)
	#train_eval(FLAGS.root_dir, 'Controller-All-v0', num_iterations=FLAGS.num_iterations)


if __name__ == '__main__':
	flags.mark_flag_as_required('root_dir')
	app.run(main)