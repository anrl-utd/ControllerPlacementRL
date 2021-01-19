import sys
print(sys.path)
import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import categorical_q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.networks import sequential, network
from tf_agents.utils import common
from spektral.layers import GCNConv, GATConv
from spektral.utils import gcn_filter

import controller_env
import pickle
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from controller_env.envs.graph_env import ControllerEnv

tf.compat.v1.enable_v2_behavior()
#train_summary_writer = tf.compat.v2.summary.create_file_writer(
#    train_dir, flush_millis=summaries_flush_secs * 1000)
#train_summary_writer.set_as_default()
global_step = tf.compat.v1.train.get_or_create_global_step()

#env_name = "CartPole-v1" # @param {type:"string"}
env_name = 'Controller-Cluster-v0'
num_iterations = 100000 # @param {type:"integer"}

initial_collect_steps = 1000  # @param {type:"integer"} 
collect_steps_per_iteration = 6  # @param {type:"integer"}
replay_buffer_capacity = 100000  # @param {type:"integer"}

fc_layer_params = (64,64,)

batch_size = 32  # @param {type:"integer"}
learning_rate = 1e-3  # @param {type:"number"}
gamma = 0.9
log_interval = 200  # @param {type:"integer"}

num_atoms = 51  # @param {type:"integer"}
min_q_value = 0  # @param {type:"integer"}
max_q_value = 30  # @param {type:"integer"}
n_step_update = 6  # @param {type:"integer"}

num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = 1000  # @param {type:"integer"}

clusters = pickle.load(open('clusters.pickle', 'rb'))
graph = nx.read_gpickle('graph.gpickle')



train_py_env = suite_gym.load(env_name, gym_kwargs={
        'graph': graph,
        'clusters': clusters})
eval_py_env = suite_gym.load(env_name, gym_kwargs={
        'graph': graph,
        'clusters': clusters})

train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

#categorical_q_net = categorical_q_network.CategoricalQNetwork(
#    train_env.observation_spec(),
#    train_env.action_spec(),
#    num_atoms=num_atoms,
#    fc_layer_params=fc_layer_params)

#optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)


#agent = categorical_dqn_agent.CategoricalDqnAgent(
#    train_env.time_step_spec(),
#    train_env.action_spec(),
#    categorical_q_network=categorical_q_net,
#    optimizer=optimizer,
#    min_q_value=min_q_value,
#    max_q_value=max_q_value,
#    n_step_update=n_step_update,
#    td_errors_loss_fn=common.element_wise_squared_loss,
#    gamma=gamma,
#    train_step_counter=global_step)
#agent.initialize()
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
        #self.dropout_1 = tf.keras.layers.Dropout(0.5)
        self.dense_1 = tf.keras.layers.Dense(30)
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
        tf_adj = tf.repeat(self.tf_gcn_adj, tf.shape(inputs)[0], axis=0)
        #tf.print(inputs, output_stream=sys.stdout)
        #tf.print(tf.reduce_sum(inputs), output_stream=sys.stdout)
        #print(tf.reduce_sum(inputs).numpy())
        #features = tf.cast(inputs, tf.float32)
        #action_out = tf.keras.layers.Flatten()(features)
        out = self.graph_conv_1([inputs, tf_adj])
        #out = self.dropout_1(out)
        out = self.graph_conv_3([out, tf_adj])
        #out = self.graph_gat_2([out, tf_adj])
        #out = self.graph_gat_1([inputs, self.tf_adj])
        #tf.print("CONV {}".format(out), output_stream=sys.stdout)
        #out = self.graph_gat_2([inputs, self.tf_adj])
        #print(out)
        #out = tf.keras.layers.Flatten()(out)
        out = self.flatten(out)
        q_out = self.dense_1(out)
        return q_out, network_state


q_net = GATNetwork(train_env.observation_spec(), train_env.action_spec(), graph)

agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    epsilon_greedy=0.1,
    n_step_update=n_step_update,
    target_update_tau=0.05,
    target_update_period=5,
    optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate),
    td_errors_loss_fn=common.element_wise_squared_loss,
    gamma=gamma,
    reward_scale_factor=1.0,
    gradient_clipping=None,
    debug_summaries=False,
    summarize_grads_and_vars=False,
    train_step_counter=global_step)
agent.initialize()

def compute_avg_return(environment, policy, num_episodes=10):

  total_return = 0.0
  for _ in range(num_episodes):

    time_step = environment.reset()
    episode_return = 0.0

    while not time_step.is_last():
      action_step = policy.action(time_step)
      time_step = environment.step(action_step.action)
      episode_return += time_step.reward
    total_return += episode_return

  avg_return = total_return / num_episodes
  return avg_return.numpy()[0]


random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                train_env.action_spec())

print(compute_avg_return(eval_env, random_policy, num_eval_episodes))

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_capacity)

def collect_step(environment, policy):
  time_step = environment.current_time_step()
  action_step = policy.action(time_step)
  next_time_step = environment.step(action_step.action)
  traj = trajectory.from_transition(time_step, action_step, next_time_step)

  # Add trajectory to the replay buffer
  replay_buffer.add_batch(traj)

for _ in range(initial_collect_steps):
  collect_step(train_env, random_policy)

# This loop is so common in RL, that we provide standard implementations of
# these. For more details see the drivers module.

# Dataset generates trajectories with shape [BxTx...] where
# T = n_step_update + 1.
dataset = replay_buffer.as_dataset(
    num_parallel_calls=3, sample_batch_size=batch_size,
    num_steps=n_step_update + 1).prefetch(3)

iterator = iter(dataset)

# (Optional) Optimize by wrapping some of the code in a graph using TF function.
agent.train = common.function(agent.train)

# Reset the train step
agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
returns = [avg_return]

for _ in range(num_iterations):

  # Collect a few steps using collect_policy and save to the replay buffer.
  for _ in range(collect_steps_per_iteration):
    collect_step(train_env, agent.collect_policy)

  # Sample a batch of data from the buffer and update the agent's network.
  experience, unused_info = next(iterator)
  train_loss = agent.train(experience)

  step = agent.train_step_counter.numpy()

  if step % log_interval == 0:
    print('step = {0}: loss = {1}'.format(step, train_loss.loss))

  if step % eval_interval == 0:
    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    print('step = {0}: Average Return = {1:.2f}'.format(step, avg_return))
    returns.append(avg_return)

steps = range(0, num_iterations + 1, eval_interval)
plt.plot(steps, returns)
plt.ylabel('Average Return')
plt.xlabel('Step')
plt.ylim(top=550)
plt.show()

print(train_env.envs[0]._gym_env.best_controllers)
print(train_env.envs[0]._gym_env.best_reward)
for i in range(5):
    train_env.envs[0]._gym_env.reset()
    centroid_controllers, heuristic_distance = train_env.envs[0]._gym_env.graphCentroidAction()
    # Convert heuristic controllers to actual
    print(centroid_controllers)
    # Assume all clusters same length
    centroid_controllers.sort()
    cluster_len = len(clusters[0])
    for i in range(len(clusters)):
        centroid_controllers[i] -= i * cluster_len
    print(centroid_controllers)
    for cont in centroid_controllers:
        (_, reward_final, _, _) = train_env.envs[0]._gym_env.step(cont)
    best_heuristic = reward_final
    print(train_env.envs[0]._gym_env.controllers, reward_final)