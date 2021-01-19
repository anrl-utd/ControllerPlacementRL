import numpy as np
from gym import spaces
import networkx as nx
from controller_env.envs.graph_env import ControllerEnv

class ControllerSingleSelect(ControllerEnv):
	"""
	Environment that selects nodes one-at-a-time directly via index. In other words,
	action is index of next node to select.
	Input: Graph with clusters
	Action: Index of next node to be controller
	State: List of whether each node is controller or not
	Reward: ControllerEnv reward
	"""
	def __init__(self, graph, clusters, pos=None):
		"""Initializes environment with no controllers"""
		super().__init__(graph, clusters, pos, check_controller_num=False)
		self.controllers = []
		self.best_controllers = []
		self.best_reward = 1e5
		self.cumulative_reward = 0
		self.num_clusters = len(clusters)
		self.action_space = spaces.Discrete(len(graph.nodes))
		self.observation_space = spaces.Box(0, 1, (len(graph.nodes),), dtype=np.int32)
		self.state = np.zeros(shape=len(graph.nodes))

	def step(self, action):
		"""
		Steps environment once
		Args:
			action: Action to perform
					Index of node to be set as controller (0-<# of nodes>)
		Returns:
			Tuple of (State, Reward) after selecting controllers and passing to base environment
			State is boolean array of size <# of nodes> which indicates whether node is controller
		"""
		self.controllers.append(action)
		self.state[action] = 1
		(self.obs, self.rew, self.done, i) = (self.state.copy(), super().step(self.controllers), len(self.controllers) >= self.num_clusters, {})
		if self.done:
			self.cumulative_reward += self.rew  # Add cumulative only when done
			if self.best_reward > self.rew:
				self.best_controllers = self.controllers
				self.best_reward = self.rew
		return (self.obs, -self.rew, self.done, i)

	def reset(self):
		"""Resets environment"""
		assert np.sum(self.state) <= 6, "Wrong number of controllers on reset - {}, {}\n{}".format(self.num_clusters, np.sum(self.state), self.state)
		assert len(self.controllers) <= 6, "Wrong number of controllers in list on reset"
		super().reset()
		self.controllers = []
		self.state = np.zeros(shape=len(self.graph.nodes))
		return self.state

class ControllerAllSelect(ControllerEnv):
	"""
	Environment that selects nodes all-at-once directly. In other words,
	action is probablities of all nodes to be controller and argmax for controllers.
	Input: Graph with clusters
	Action: Probabilities of node to be controller
	State: List of whether each node is controller or not
	Reward: ControllerEnv reward
	"""
	def __init__(self, graph, clusters, pos=None):
		"""Initializes environment with no controllers"""
		super().__init__(graph, clusters, pos, check_controller_num=True)
		self.controllers = []
		self.best_controllers = []
		self.best_reward = 1e5
		self.cumulative_reward = 0
		self.num_clusters = len(clusters)
		self.action_space = spaces.Box(0, 1, (len(graph.nodes),), dtype=np.float32)
		self.observation_space = spaces.Box(0, 1, (len(graph.nodes),), dtype=np.int32)
		self.state = np.zeros(shape=len(graph.nodes))

	def step(self, action):
		"""
		Steps environment once
		Args:
			action: Action to perform
					Index of node to be set as controller (0-<# of nodes>)
		Returns:
			Tuple of (State, Reward) after selecting controllers and passing to base environment
			State is boolean array of size <# of nodes> which indicates whether node is controller
		"""
		self.controllers = np.argpartition(action, -self.num_clusters)[-self.num_clusters:]
		np.put(self.state, self.controllers, [1])
		assert np.sum(self.state) == self.num_clusters
		(self.obs, self.rew, self.done, i) = (self.state.copy(), super().step(self.controllers), len(self.controllers) >= self.num_clusters, {})
		if self.done:
			self.cumulative_reward += self.rew  # Add cumulative only when done
			if self.best_reward > self.rew:
				self.best_controllers = self.controllers
				self.best_reward = self.rew
		return (self.obs, -self.rew, self.done, i)

	def reset(self):
		"""Resets environment"""
		assert np.sum(self.state) <= 6, "Wrong number of controllers on reset - {}, {}\n{}".format(self.num_clusters, np.sum(self.state), self.state)
		assert len(self.controllers) <= 6, "Wrong number of controllers in list on reset"
		super().reset()
		self.controllers = []
		self.state = np.zeros(shape=len(self.graph.nodes))
		return self.state