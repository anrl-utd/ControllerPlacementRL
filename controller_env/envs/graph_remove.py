"""
Environment for reinforcement learning
"""
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import warnings
import random
import itertools
import pprint
from scipy.interpolate import interp1d
from collections import defaultdict
import heapq
import time

warnings.filterwarnings("ignore", category=UserWarning)
random.seed(0)
np.random.seed(3)



class ControllerRemove(gym.Env):
    """Base environment used to simulate the network for the RL"""
    metadata = {'render.modes': ['human']}

    def __init__(self, graph, clusters, bridge_nodes, pos=None):
        """Initializes the environment (Runs at first)"""
        print("Initialized environment!")
        self.action_space = spaces.Box(np.zeros(len(clusters)), np.ones(len(clusters)) * len(graph.nodes),
                                       dtype=np.uint8)
        # self.observation_space = spaces.Box(np.zeros(shape=len(graph.nodes)), np.ones(shape=len(graph.nodes)), dtype=np.bool)
        self.original_graph = graph.copy()
        if (pos is None):
            self.pos = nx.spring_layout(graph)  # get the positions of the nodes of the graph
        else:
            self.pos = pos
        self.clusters = np.stack(clusters)
        self.graph = graph.copy()
        # self.degree = self._graph_degree()
        self.bridge_nodes = bridge_nodes

    def calculateOptimal(self):
        combinations = list(itertools.product(*self.clusters))
        min_dist = 1000000
        min_combination = None
        for combination in combinations:
            dist = self.calculateDistance(combination)
            if (dist < min_dist):
                min_dist = dist
                min_combination = combination
        return (min_combination, min_dist)

    def calculateDistance(self, actions):
        totalDist = 0
        for action in list(itertools.combinations(actions, 2)):
            # print("distance to find is: ", action)
            # print(nx.shortest_path_length(self.graph, action[0], action[1], weight='weight'))
            totalDist += nx.shortest_path_length(self.graph, action[0], action[1], weight = 'weight')
        return totalDist
    def returnGraph(self):
        return self.graph
    def showRemainingControllers(self):
        return (self.graph.nodes, self.calculateDistance(self.graph.nodes))
    def reset(self):
        print("Environment reset")
        self.graph = self.original_graph

    # def step(self, action, num_packets=100):
    #     """Steps the environment once"""
    #     """
    #     How it works:
    #     action is indices of controllers
    #     Create a new complete graph with controllers only
    #      - Use shortest-distance between controllers as the weight of edges for controller graph
    #      - Store indices of nodes under each controller
    #     Create several "packets" with source and destination
    #     Find controller for source and destination
    #     If source is same as destination controller, distance is 0
    #     Otherwise, distance is the shortest-path distance between the source controller and destination
    #     Add up all distances and have that as initial reward
    #     """
    #     distance = 0
    #     controller_graph = None
    #     # Create metagraph of controllers. The node at an index corresponds to the controller of the cluster of that index
    #     try:
    #         controller_graph = self._set_controllers(action)
    #         """
    #         Don't create packets, try having reward of total distance between all adjacent controllers
    #
    #         #Create "packets" with source, destination
    #         packets = np.random.randint(low=0, high=len(self.graph.nodes), size=(num_packets, 2))
    #         #Convert source and destination to cluster the source/destination is in
    #         for i in range(packets.shape[0]):
    #             if packets[i, 0] == packets[i, 1]:
    #                 continue
    #             source_cluster = np.where(self.clusters == packets[i, 0])[0][0]
    #             destination_cluster = np.where(self.clusters == packets[i, 1])[0][0]
    #             distance += nx.dijkstra_path_length(controller_graph, source_cluster, destination_cluster)
    #         """
    #     except AssertionError:
    #         return 100000
    #     # Return output reward
    #     # return -distance
    #     return controller_graph.size(weight='weight')

    # def reset(self):
    #     """Resets the environment to initial state"""
    #     print("Reset environment")

    def render(self, mode='human', close=False):
        """Renders the environment once"""
        plt.clf()  # Clear the matplotlib figure

        # Redraw the entire graph (this can only be expedited if we save the position and colors beforehand)
        # Then we won't have to recalculate all this to draw. Maybe make them a global variable?
        node_colors = list(self.graph)
        clustering = nx.get_node_attributes(self.graph, 'cluster')
        for index, node in enumerate(list(node_colors)):
            node_colors[index] = clustering[node]
        nx.draw_networkx_nodes(self.graph, self.pos, node_color=node_colors)
        nx.draw_networkx_edges(self.graph, self.pos, self.graph.edges())  # draw the edges of the self.graph
        nx.draw_networkx_labels(self.graph, self.pos)  # draw  the labels of the self.graph
        edge_labels = nx.get_edge_attributes(self.graph, 'weight')
        nx.draw_networkx_edge_labels(self.graph, self.pos,
                                     edge_labels=edge_labels)  # draw the edge weights of the self.graph
        plt.draw()
        plt.show()



    def renderCentroid(self, controllers, mode='human', close=False):
        """Renders the environment once"""
        plt.clf()  # Clear the matplotlib figure
        graph_centroid = self.findGraphCentroid()
        # Redraw the entire graph (this can only be expedited if we save the position and colors beforehand)
        # Then we won't have to recalculate all this to draw. Maybe make them a global variable?
        node_colors = list(self.graph)
        node_size = list(self.graph)
        clustering = nx.get_node_attributes(self.graph, 'cluster')
        for index, node in enumerate(list(node_colors)):
            node_colors[index] = clustering[node]
        # node_colors[controllers] = len(self.clusters) + 2
        # for controller in controllers:
        #     node_colors[controller] = len(self.clusters) + 2
        node_colors[graph_centroid[0]] = len(self.clusters) + 1
        for index in range(len(node_size)):
            if index not in controllers and index != graph_centroid[0]:
                node_size[index] = 300
            else:
                node_size[index] = 1000
        edge_color = list(range(len(self.graph.edges)))
        shortest_paths = []
        for controller in controllers:
            shortest_path = nx.shortest_path(self.graph, controller, graph_centroid[0], weight = 'weight')
            shortest_paths = shortest_paths + [(min(shortest_path[index], shortest_path[index + 1]), max(shortest_path[index], shortest_path[index + 1])) for index in range(len(shortest_path) -1)]
        for index in range(len(edge_color)):
            edge_color[index] = 1
        for index in range(len(shortest_paths)):
            edge_color[list(self.graph.edges()).index(shortest_paths[index])] = 2

        nx.draw_networkx_nodes(self.graph, self.pos, node_color=node_colors, node_size = node_size)
        nx.draw_networkx_edges(self.graph, self.pos, self.graph.edges(), edge_color = edge_color)  # draw the edges of the self.graph
        nx.draw_networkx_labels(self.graph, self.pos)  # draw  the labels of the self.graph
        edge_labels = nx.get_edge_attributes(self.graph, 'weight')
        nx.draw_networkx_edge_labels(self.graph, self.pos,
                                     edge_labels=edge_labels)  # draw the edge weights of the self.graph
        plt.draw()
        plt.show()

    def decrementCluster(self):
        pass

    def sigmoid(self, a):
        return (1 / (1 + np.exp(-a)))

    def step(self, ):
        actions = []
        translator = interp1d([0, 7100], [0, 10])
        for index in range(0, len(self.clusters)):
            twoRandomNodes = [np.random.choice(self.clusters[index], 2, replace = False)]
            randomOneLengthToBridge = 0
            randomTwoLengthToBridge = 0
            for index2 in self.clusters[index]:
                if self.graph.nodes[index2]['bridge_node'] == True:

                    if index2 == twoRandomNodes[0][0]:
                            pass#randomOneLengthToBridge += self.graph.nodes[index2]['bridge_weight']
                    else:
                        print("-----------------------------", translator(self.graph.nodes[index2]['bridge_weight']))
                        randomOneLengthToBridge += nx.shortest_path_length(self.graph, twoRandomNodes[0][0], index2, weight="weight") * (1 + translator(self.graph.nodes[index2]['bridge_weight']))
                    if index2 == twoRandomNodes[0][1]:
                        pass# randomTwoLengthToBridge += self.graph.nodes[index2]['bridge_weight']
                    else:
                        print("-----------------------------", translator(self.graph.nodes[index2]['bridge_weight']))
                        randomTwoLengthToBridge += nx.shortest_path_length(self.graph, twoRandomNodes[0][1], index2, weight="weight") * (1 + translator(self.graph.nodes[index2]['bridge_weight']))

            actions.append(twoRandomNodes[0][0] if randomOneLengthToBridge > randomTwoLengthToBridge else twoRandomNodes[0][1])
            print("two nodes considering removing is: ", twoRandomNodes)
        # actions = [np.random.choice(self.clusters[0]), np.random.choice(self.clusters[1]), np.random.choice(self.clusters[2]), np.random.choice(self.clusters[3]), np.random.choice(self.clusters[4]), np.random.choice(self.clusters[5])]
        print("The actions to remove are: ", actions)
        for action in actions:
            edges = []
            for _, b in enumerate(self.graph[action].items()):
                cluster_bridges = {} #variable to hold shortest path between clusters
                if len(edges) == 0:
                    edges.append((b[0], b[1]['weight']))
                    continue
                for edge in edges:
                    if b[0] in self.graph[edge[0]]:
                        self.graph.add_edge(edge[0], b[0] ,weight = min(self.graph[edge[0]][b[0]]['weight'], self.graph[action][b[0]]['weight'] + self.graph[action][edge[0]]['weight']))
                        # if self.graph.nodes[b[0]]['cluster'] != self.graph.nodes[action]['cluster']:
                        #     if cluster_bridges[self.graph.nodes[b[0]]['cluster']] != None:
                        #         if self.graph[edge[0]][b[0]]['weight'] < cluster_bridges[self.graph.nodes[b[0]]['cluster']][1]:
                        #             cluster_bridges[self.graph.nodes[b[0]]['cluster']] = (edge[0], self.graph[edge[0]][b[0]]['weight'])
                        #         else:
                        #             cluster_bridges[self.graph.nodes[b[0]]['cluster']] = (edge[0], self.graph[edge[0]][b[0]]['weight'])
                    else:
                        self.graph.add_edge(edge[0], b[0], weight = self.graph[action][b[0]]['weight'] + self.graph[action][edge[0]]['weight'])
                        # if self.graph.nodes[b[0]]['cluster'] != self.graph.nodes[action]['cluster']:
                        #     if cluster_bridges[self.graph.nodes[b[0]]['cluster']] != None:
                        #         if self.graph[edge[0]][b[0]]['weight'] < cluster_bridges[self.graph.nodes[b[0]]['cluster']]:
                        #             cluster_bridges[self.graph.nodes[b[0]]['cluster']] = self.graph[edge[0]][b[0]]['weight']
                        #         else:
                        #             cluster_bridges[self.graph.nodes[b[0]]['cluster']] = self.graph[edge[0]][b[0]]['weight']
                        # self.graph[edge[0]][b[0]]['weight'] = self.graph[action][b[0]]['weight'] + self.graph[action][edge[0]]['weight']
                # if self.graph.nodes[action]['bridge_node'] == 1:
                for index, item in enumerate(cluster_bridges):
                    self.graph.nodes[b[0]]['bridge_node'] = 1
                edges.append((b[0], b[1]['weight']))


        self.graph.remove_nodes_from(actions)
        new_clusters = np.zeros((len(self.clusters), len(self.clusters[0]) - 1), np.uint8)
        for index in range(len(self.clusters)):
            new_clusters[index] = np.delete(self.clusters[index], list(self.clusters[index]).index(actions[index]))
        self.clusters = np.stack(new_clusters)
        self.pos = nx.spring_layout(self.graph)  # get the positions of the nodes of the graph
        if len(list(self.graph.nodes)) == len(self.clusters):
            return True
        return False





    # def _set_controllers(self, controllers):
    #     """Creates metagraph of controllers
    #     Args:
    #         controllers: Array of controller indices
    #
    #     Returns:
    #         Complete graph of controllers (metagraph)
    #
    #     Raises:
    #         AssertError: Issue with controller indices (not 1 per cluster)"""
    #     # Ensure that these are valid controllers - all clusters have a controller
    #     assert (len(controllers) == self.clusters.shape[0])
    #     found_clusters = np.zeros((len(controllers)))  # Stores what clusters have controllers been found for
    #     clusters = nx.get_node_attributes(self.graph, 'cluster')
    #     index = 0
    #     for controller in controllers:
    #         # Multiple controllers in a cluster
    #         assert (found_clusters[clusters[controller]] == 0)
    #         found_clusters[clusters[controller]] = 1
    #
    #     # Controllers were found to be valid. Now add controllers to complete metagraph.
    #     # TODO: Optimize this, new_contr_indices and mapping can be reduced to a single variable (and possible a single line for the for)
    #     new_contr_indices = []
    #     # mapping = defaultdict(list)
    #     for i in range(len(controllers)):
    #         new_contr_indices.append([i, controllers[i]])
    #     # mapping[i] = controllers[i]
    #     controller_graph = nx.complete_graph(len(new_contr_indices))  # Store controller metagraph
    #
    #     for pair in itertools.combinations(new_contr_indices, 2):
    #         controller_graph.add_edge(pair[0][0], pair[1][0],
    #                                   weight=nx.dijkstra_path_length(self.graph, source=pair[0][1], target=pair[1][1]))
    #
    #     ##Display metagraph for debugging. Should be removed once we get _set_controllers() working
    #     # display_graph = nx.relabel_nodes(controller_graph, mapping)
    #     ## nx.draw_networkx_nodes(display_graph,self. pos)
    #     # nx.draw_networkx_edges(display_graph, self.pos, display_graph.edges())        # draw the edges of the display_graph
    #     # nx.draw_networkx_labels(display_graph, self.pos)                      # draw  the labels of the display_graph
    #     # edge_labels = nx.get_edge_attributes(display_graph,'weight')
    #     # nx.draw_networkx_edge_labels(display_graph,self.pos,edge_labels=edge_labels) # draw the edge weights of the display_graph
    #     # plt.draw()
    #     # plt.show()
    #
    #     return controller_graph

    # def _graph_degree(self):
    #     """Returns the highest degree of a node in the graph"""
    #     return max([degree for node, degree in self.graph.degree()])

    # def _random_valid_controllers(self):
    #     """Intended for testing, this gives a random set of valid controllers"""
    #     cluster_arr = np.asarray(self.graph.nodes.data('cluster'))  # Creates NumPy array with [node #, cluster #] rows
    #     controller_indices = []
    #     for cluster in range(self.clusters.shape[0]):  # For every cluster
    #         cluster_controller = np.random.choice(cluster_arr[cluster_arr[:, 1] == cluster][:,
    #                                               0])  # Select all nodes of a cluster then choose one randomly
    #         controller_indices.append(cluster_controller)
    #     return controller_indices

    # def stepLA(self, graph, controllers):
    #     """
    #     Helper function used to calculate distance between chosen controllers
    #     :param controllers: The list of chosen controllers in which to calculate distance between
    #     :return: The total distance from every controller to the other controllers
    #     """
    #     distance = 0
    #     for current_controller in range(len(controllers)):
    #         for other_controller in range(current_controller, len(controllers)):
    #             distance += nx.dijkstra_path_length(graph, controllers[current_controller],
    #                                                 controllers[other_controller])
    #     return distance

    # def calculateOptimal(self):
    #     combinations = list(itertools.product(*self.clusters))
    #     min_dist = 1000000
    #     min_combination = None
    #     for combination in combinations:
    #         dist = self.step(combination)
    #         if (dist < min_dist):
    #             min_dist = dist
    #             min_combination = combination
    #     return (min_combination, min_dist)
    def pickAction(self):
        nodesRemoved = []
        for cluster in self.clusters():
            np.random.choice(cluster)

    def showSubgraph(self, cluster_num):
        self.minimumSpanningTree(self.clusters[cluster_num])
    def minimumSpanningTree(self,subtree_nodes):
        subgraph =  nx.minimum_spanning_tree(self.graph.subgraph(subtree_nodes))
        print(subgraph.edges())
        """Renders the environment once"""
        plt.clf()  # Clear the matplotlib figure

        # Redraw the entire graph (this can only be expedited if we save the position and colors beforehand)
        # Then we won't have to recalculate all this to draw. Maybe make them a global variable?
        node_colors = list(subgraph)
        clustering = nx.get_node_attributes(subgraph, 'cluster')
        for index, node in enumerate(list(node_colors)):
            node_colors[index] = clustering[node]
        nx.draw_networkx_nodes(subgraph, nx.spring_layout(subgraph), node_color=node_colors)
        nx.draw_networkx_edges(subgraph, nx.spring_layout(subgraph), subgraph.edges())  # draw the edges of the self.graph
        nx.draw_networkx_labels(subgraph, nx.spring_layout(subgraph))  # draw  the labels of the self.graph
        edge_labels = nx.get_edge_attributes(subgraph, 'weight')
        nx.draw_networkx_edge_labels(subgraph, nx.spring_layout(subgraph),
                                     edge_labels=edge_labels)  # draw the edge weights of the self.graph
        plt.draw()
        plt.show()

    def recalculateBridgeWeight(self):
        graph_centroid = self.findGraphCentroid()[0]
        for node in self.graph.nodes:
            self.graph.nodes[node]['bridge_weight'] = 0
            if self.graph.nodes[node]['bridge_node'] ==1 :
                self.graph.nodes[node]['bridge_weight'] = nx.shortest_path_length(self.graph, node, graph_centroid, weight='weight')
                # for k, v in enumerate(self.graph[node].items()):
                #     if self.graph.nodes[v[0]]['bridge_node'] == 1 and self.graph.nodes[v[0]]['cluster'] != self.graph.nodes[node]['cluster']:
                #         self.graph.nodes[node]['bridge_weight'] += v[1]['weight']

    def calculateSpanningOptimal(self):
        answers = [] # the output of the best controllers
        translator = interp1d([-1.1, 100], [1, 1.2])
        for cluster in range(0, len(self.clusters)): # for each of the clusters
            for bridge_index in range(len(self.bridge_nodes[cluster])):  # for each bridge node in the current cluster being traversed
                queue = []  # variable to show the remaining nodes in the cluster being traversed
                # nx.set_node_attributes(self.graph, 'current_distance', 0) # resets all nodes current distance attribute to zero
                for node in self.clusters[cluster]:
                    self.graph.nodes[node]['current_distance'] = 0
                self.graph.nodes[self.bridge_nodes[cluster][bridge_index][0]]['current_distance'] = 0.0000000000000001 #self.bridge_nodes[cluster][bridge_index][1]  # set the bridge node being traversed's current distance to its edge weight to anothnr cluster
                queue.append(self.bridge_nodes[cluster][bridge_index][0]) # append the bridge node to start off the traversing from that bridge node
                while len(queue) != 0: # while the cluster has not been traversed
                    currentNodeToExtractDistance = queue.pop(0) # pop out the next node to propagate distnace from
                    for _, neighbor in enumerate(self.graph[currentNodeToExtractDistance].items()): # for each neighbor of the current node being used to propagate information from
                        if self.graph.nodes[neighbor[0]]['current_distance'] != 0: # if the neighbor has already been propagated information from
                            continue # skip this curernt iteration
                        print(self.graph.nodes[neighbor[0]]['cluster'])
                        if self.graph.nodes[neighbor[0]]['cluster'] != self.graph.nodes[currentNodeToExtractDistance]['cluster']:
                            continue
                        self.graph.nodes[neighbor[0]]['current_distance'] = self.graph.nodes[currentNodeToExtractDistance]['current_distance'] + self.graph[neighbor[0]][currentNodeToExtractDistance]['weight'] # update this neighbor's distance to bridge node
                        queue.append(neighbor[0]) # append this neighbor

                for node in self.clusters[cluster]:
                    # if self.graph.nodes[node]['current_distance'] == 0.0000000000000001:
                        # self.graph.nodes[node]['current_distance'] = 0
                    self.graph.nodes[node]['total_distance'] += self.graph.nodes[node]['current_distance'] * self.bridge_nodes[cluster][bridge_index][1]
            lowestValue = 100000000000
            bestIndex = -1
            for index in self.clusters[cluster]:
                value = self.graph.nodes[index]['total_distance']
                if value < lowestValue and value != 0:
                    lowestValue = value
                    bestIndex = index
            answers.append(bestIndex)
        return answers


    def calculateNewOptimal(self):
        answers = [] # the output of the best controllers
        translator = interp1d([-1.1, 100], [1, 1.2])
        self.recalculateBridgeWeight()
        for cluster in range(0, len(self.clusters)): # for each of the clusters
            for bridge_index in range(len(self.bridge_nodes[cluster])):  # for each bridge node in the current cluster being traversed
                queue = []  # variable to show the remaining nodes in the cluster being traversed
                # queue = heapq.heapify(queue)
                # nx.set_node_attributes(self.graph, 'current_distance', 0) # resets all nodes current distance attribute to zero
                for node in self.clusters[cluster]:
                    self.graph.nodes[node]['current_distance'] = 0
                    self.graph.nodes[node]['visited'] = False
                self.graph.nodes[self.bridge_nodes[cluster][bridge_index][0]]['current_distance'] = 0.0000000000000001 #self.bridge_nodes[cluster][bridge_index][1]  # set the bridge node being traversed's current distance to its edge weight to anothnr cluster
                heapq.heappush(queue, (0, self.bridge_nodes[cluster][bridge_index][0])) # append the bridge node to start off the traversing from that bridge node,  the 0 doesn't matter
                while len(queue) != 0: # while the cluster has not been traversed
                    currentNodeToExtractDistance = heapq.heappop(queue)[1] # pop out the next node to propagate distnace from
                    # TODO Fix for efficientcy as currently storing duplicates in queue
                    if self.graph.nodes[currentNodeToExtractDistance]['visited'] == True:
                        continue
                    self.graph.nodes[currentNodeToExtractDistance]['visited'] = True
                    for _, neighbor in enumerate(self.graph[currentNodeToExtractDistance].items()): # for each neighbor of the current node being used to propagate information from
                        if self.graph.nodes[neighbor[0]]['visited'] == True: # if the neighbor has already been propagated information from
                            continue # skip this curernt iteration
                        if self.graph.nodes[neighbor[0]]['cluster'] != self.graph.nodes[currentNodeToExtractDistance]['cluster']:
                            continue
                        self.graph.nodes[neighbor[0]]['current_distance'] = min(self.graph.nodes[neighbor[0]]['current_distance'], self.graph.nodes[currentNodeToExtractDistance]['current_distance'] + self.graph[neighbor[0]][currentNodeToExtractDistance]['weight']) if self.graph.nodes[neighbor[0]]['current_distance'] != 0 else (self.graph.nodes[currentNodeToExtractDistance]['current_distance'] + self.graph[neighbor[0]][currentNodeToExtractDistance]['weight'])# update this neighbor's distance to bridge node
                        heapq.heappush(queue, (self.graph.nodes[neighbor[0]]['current_distance'], neighbor[0])) # append this neighbor

                for node in self.clusters[cluster]:
                    # if self.graph.nodes[node]['current_distance'] == 0.0000000000000001:
                        # self.graph.nodes[node]['current_distance'] = 0
                    self.graph.nodes[node]['total_distance'] += self.graph.nodes[node]['current_distance'] / self.graph.nodes[self.bridge_nodes[cluster][bridge_index][0]]['bridge_weight'] if self.graph.nodes[self.bridge_nodes[cluster][bridge_index][0]]['bridge_weight'] != 0 else self.graph.nodes[node]['current_distance'] / 0.000000001
            lowestValue = 100000000000
            bestIndex = -1
            for index in self.clusters[cluster]:
                value = self.graph.nodes[index]['total_distance']
                if value < lowestValue and value != 0:
                    lowestValue = value
                    bestIndex = index
            answers.append(bestIndex)
        return answers

    def findGraphCentroid(self):
        lowest_weight = 100000000
        best_node = -1
        for cur_node in self.graph.nodes:
            cur_weight = 0
            for other_node in self.graph.nodes:
                if other_node == cur_node:
                    continue
                cur_weight += nx.shortest_path_length(self.graph, cur_node, other_node, weight = 'weight')
            # print("This is the length to all other nodes:", cur_weight, cur_node)
            if cur_weight < lowest_weight:
                lowest_weight = cur_weight
                best_node = cur_node
        return best_node, lowest_weight

    def best_action(self):
        actions = []
        centroid = self.findGraphCentroid()[0]
        for index, cluster in enumerate(self.clusters):
            if self.graph.nodes[centroid]['cluster'] == index:
                continue
            bestNode = None
            lowestDistance = 100000000
            for node in cluster:
                if nx.shortest_path_length(self.graph, centroid, node, weight = 'weight') < lowestDistance:
                    lowestDistance = nx.shortest_path_length(self.graph, centroid, node, weight = 'weight')
                    bestNode = node
            actions.append(bestNode)
        bestNode = None
        lowestDistance = 10000000
        for node in self.clusters[self.graph.nodes[centroid]['cluster']]:
            if self.calculateDistance(actions + [node]) < lowestDistance:
                lowestDistance = self.calculateDistance(actions + [node])
                bestNode = node
        actions.append(bestNode)

        t0 = time.time()
        # Simulated Annealing Meta Heuristic
        current_state = actions
        annealing_rate = 0.90
        for x in range(1000):
            temperature = 1
            randomCluster = np.random.randint(len(self.clusters))
            randomBestAction = current_state[randomCluster]
            neighborList = [v for k, v in self.graph.edges if k == randomBestAction and v in self.clusters[self.graph.nodes[randomBestAction]['cluster']]]
            neighborList.append(randomBestAction)
            randomNeighbor = np.random.choice(neighborList)
            proposed_state = current_state.copy()
            proposed_state[randomCluster] = randomNeighbor
            threshold = float(np.random.rand(1))
            proposed_distance = self.calculateDistance(proposed_state)
            current_distance = self.calculateDistance(current_state)
            probability = 1 if proposed_distance < current_distance else np.exp(-(proposed_distance - current_distance)/temperature)
            if probability >= threshold:
                current_state = proposed_state.copy()
            temperature *= annealing_rate
        t1 = time.time()
        total_time = t1 - t0
        # if self.calculateDistance(actions) < self.calculateDistance(current_state):
        #     current_state = actions
        return current_state, self.calculateDistance(current_state), actions, self.calculateDistance(actions), total_time


def generateGraph(num_clusters, num_nodes, prob_cluster=0.5, prob=0.2, weight_low=0, weight_high=100, draw=True):
    """Generates graph given number of clusters and nodes
    Args:
        num_clusters: Number of clusters
        num_nodes: Number of nodes
        prob_cluster: Probability of adding edge between any two nodes within a cluster
        prob: Probability of adding edge between any two nodes
        weight_low: Lowest possible weight for edge in graph
        weight_high: Highest possible weight for edge in graph
        draw: Whether or not to show graph (True indicates to show)

    Returns:
        Graph with nodes in clusters, array of clusters, graph position for drawing
    """
    node_colors = np.arange(0, num_nodes, 1, np.uint8)  # Stores color of nodes
    G = nx.Graph()
    node_num = 0
    nodes_per_cluster = int(num_nodes / num_clusters)
    clusters = np.zeros((num_clusters, nodes_per_cluster), np.uint8)  # Stores nodes in each cluster

    # Create clusters and add random edges within each cluster before merging them into single graph
    for i in range(num_clusters):
        # Add tree to serve as base of cluster subgraph. Loop through all edges and assign weights to each
        cluster = nx.random_tree(nodes_per_cluster)
        for start, end in cluster.edges:
            cluster.add_edge(start, end, weight=random.randint(weight_low, weight_high))

        # Add edges to increase connectivity of cluster
        new_edges = np.random.randint(0, nodes_per_cluster, (int(nodes_per_cluster * prob_cluster), 2))
        new_weights = np.random.randint(weight_low, weight_high, (new_edges.shape[0], 1))
        new_edges = np.append(new_edges, new_weights, 1)
        cluster.add_weighted_edges_from(new_edges)

        # Set attributes and colors
        nx.set_node_attributes(cluster, i, 'cluster')
        nx.set_node_attributes(cluster, 0.5, 'learning_automation')
        nx.set_node_attributes(cluster, 0, 'bridge_node')
        nx.set_node_attributes(cluster, 0, 'bridge_weight')
        nx.set_node_attributes(cluster, 0, 'current_distance')
        nx.set_node_attributes(cluster, 0, 'total_distance')
        nx.set_node_attributes(cluster, False, 'visited')
        node_colors[node_num:(node_num + nodes_per_cluster)] = i
        node_num += nodes_per_cluster
        clusters[i, :] = np.asarray(cluster.nodes) + nodes_per_cluster * i


        # # TO RENDER ALL THE CLUSTERS
        # plt.clf()  # Clear the matplotlib figure
        #
        # # Redraw the entire graph (this can only be expedited if we save the position and colors beforehand)
        # # Then we won't have to recalculate all this to draw. Maybe make them a global variable?
        # node_colors = list(cluster)
        # clustering = nx.get_node_attributes(cluster, 'cluster')
        # for index, node in enumerate(list(node_colors)):
        #     node_colors[index] = clustering[node]
        # nx.draw_networkx_nodes(cluster, nx.spring_layout(cluster), node_color=node_colors)
        # nx.draw_networkx_edges(cluster, nx.spring_layout(cluster), cluster.edges())  # draw the edges of the self.graph
        # nx.draw_networkx_labels(cluster, nx.spring_layout(cluster))  # draw  the labels of the self.graph
        # edge_labels = nx.get_edge_attributes(cluster, 'weight')
        # nx.draw_networkx_edge_labels(cluster, nx.spring_layout(cluster),
        #                              edge_labels=edge_labels)  # draw the edge weights of the self.graph
        # plt.draw()
        # plt.show()
        # # END OF CODEqueue OF RENDERING ALL THE CLUSTERS

        # Merge cluster with main graph
        # cluster = nx.minimum_spanning_tree(cluster)
        G = nx.disjoint_union(G, cluster)


    bridge_nodes = []
    for index in range(num_clusters):
        bridge_nodes.append([])
    # Add an edge to connect all clusters (to gurantee it is connected)
    node_num = 0
    for i in range(num_clusters - 1):
        randomAddition = np.random.randint(0, nodes_per_cluster)
        node_num += randomAddition
        G.add_edge(node_num, node_num + nodes_per_cluster, weight=random.randint(weight_low, weight_high))
        G.nodes[node_num]['bridge_node'] = 1
        bridge_nodes[G.nodes[node_num]['cluster']].append((node_num, G[node_num][node_num + nodes_per_cluster]['weight']))
        # G.nodes[node_num]['bridge_weight'] += G[node_num][node_num + nodes_per_cluster]['weight']
        G.nodes[node_num + nodes_per_cluster]['bridge_node'] = 1
        bridge_nodes[G.nodes[node_num + nodes_per_cluster]['cluster']].append((node_num + nodes_per_cluster, G[node_num][node_num + nodes_per_cluster]['weight']))
        # G.nodes[node_num + nodes_per_cluster]['bridge_weight'] += G[node_num][node_num + nodes_per_cluster]['weight']
        node_num += nodes_per_cluster
        node_num -= randomAddition

    # Add random edges to any nodes to increase diversity
    new_edges = np.random.randint(0, num_nodes, (int(num_nodes * 0.1), 2))
    for new_edge in new_edges:
        if G.nodes[new_edge[0]]['cluster'] != G.nodes[new_edge[1]]['cluster']:
            G.nodes[new_edge[0]]['bridge_node'] = 1
            G.nodes[new_edge[1]]['bridge_node'] = 1
    new_weights = np.random.randint(weight_low, weight_high, (new_edges.shape[0], 1))
    new_edges = np.append(new_edges, new_weights, 1)
    G.add_weighted_edges_from(new_edges)
    G.remove_edges_from(nx.selfloop_edges(G))  # Remove self-loops caused by adding random edges


    # Draw graph
    pos = nx.spring_layout(G)
    if draw:
        nx.draw_networkx_nodes(G, pos, node_color=node_colors)
        nx.draw_networkx_labels(G, pos)
        nx.draw_networkx_edges(G, pos, G.edges())
        plt.draw()
        plt.show()
    return G, clusters, bridge_nodes, pos

