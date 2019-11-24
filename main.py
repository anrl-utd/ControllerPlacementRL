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

graph, clusters, pos = generateGraph(5, 200, draw=False)	#Generate graph
#Nudging environment
# env = gym.make('Controller-RandomPlacement-v0', graph=graph, clusters=clusters, pos=pos)
# reward = env.step([np.random.randint(0, 2) for i in clusters])	#This throws errors every so often since it selects neighbors without distinction whether the neighbor is in the cluster or not

#Base environment
env = gym.make('Controller-v0', graph=graph, clusters=clusters, pos=pos)

best_controller_distance = math.inf   # positive infinity so that any distance in the graph will become next best distance
best_controllers = []   # list to hold the best found controllers so far
epochs = 2500   # number of iterations for the for loop
alpha = 0.09    # growth rate when changing probabilities
# amountPicks = dict.fromkeys(graph.nodes, 0)   # unused for now but to hold the amount each node became controller
totalRewards = []                    # list to hold the rewards per epoch in order to plot with matplotlib
epsilon = 0.95  # exploration rate for epsilon greedy that will go down with episolon / epoch

for epoch in range(1, epochs):
    chosenControllers = []
    # print(epoch)

    # action = [np.random.choice(i) for i in clusters]

    """
        This  section is purely for learning automaton with a custom twist 
    """
    for index in range(0, len(clusters)):
        while len(chosenControllers)-1 != index:         # while the controller for the current cluster has not been picked or it is the first controller being picked
            randomChoice = random.choice(list(clusters[index]))
            # only pick a controller if the probability that it becomes a controller is greater than a random number
            controller = randomChoice if graph.nodes.data()[randomChoice]['learning_automaton'] > random.random() else None
            if controller is not None:
                chosenControllers.append(controller)
                # amountPicks[controller] += 1

    """
        This section is using argmax on each cluster's nodes' probabilities
    """
    # for index in range(0, len(clusters)):
    #     if epsilon * ((epochs - epoch)/epochs) > random.random():
    #         chosenControllers.append(random.choice(clusters[index]))
    #     else:
    #         chosenControllers.append(max(clusters[index]))
    #      # amountPicks[controller] += 1
    # print("controllers are: " + str(chosenControllers))
    #
    # print("epsiolonis : "+ str(epsilon * ((epochs - epoch)/epochs)))

    action = chosenControllers
    # print(action)

    reward = env.stepLA(graph, action)
    if reward < best_controller_distance:
        best_controller_distance = reward
        best_controllers = chosenControllers
    totalRewards.append(reward)

    #increment or decrement probabilities of becoming controllers
    for outer_index in range(0, len(clusters)):
        for inner_index in range(0, len(clusters[outer_index])):
            cur_learn_automaton = graph.nodes.data()[clusters[outer_index][inner_index]]['learning_automaton']
            # amountPicked = amountPicks[clusters[outer_index][inner_index]] if amountPicks[clusters[outer_index][inner_index]] > 1

            # make alpha equal to upper confidence bound is an idea, may implement later for benchmarks where cluster = 5, nodes = 200, and used 50000, 5000, 2500 with all same results

            """ 
                Uses the probabilistic update described in the learning automaton paper  
            """
            # if the reward is the optimal reward so far
            if reward == best_controller_distance or reward < best_controller_distance:
                graph.nodes.data()[clusters[outer_index][inner_index]]['learning_automaton'] = cur_learn_automaton + alpha * (1 - cur_learn_automaton) if \
                clusters[outer_index][inner_index] in chosenControllers else cur_learn_automaton * (1 - alpha)
            # if the reward is suboptimal
            else:
                graph.nodes.data()[clusters[outer_index][inner_index]]['learning_automaton'] = cur_learn_automaton * (1 - alpha) if \
                clusters[outer_index][inner_index] in chosenControllers else cur_learn_automaton

    # print("REWARD:", reward)

env._set_controllers(best_controllers)


#My debugging statements:
for outer_index in range(0, len(clusters)):
    for inner_index in range(0, len(clusters[outer_index])):
        print("Probability for node " + str(clusters[outer_index][inner_index]) + ": " + str(graph.nodes.data()[clusters[outer_index][inner_index]]['learning_automaton']))
print("Best controllers are: " + str(best_controllers))
print("Best distance is: " + str(best_controller_distance))
# print("Amount of Picks: " + str(amountPicks))
plt.plot(totalRewards)
plt.show()


env.render()
env.reset()