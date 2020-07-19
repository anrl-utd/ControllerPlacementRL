def best_action(self):
    actions = []  # Set of Controllers
    centroid = self.findGraphCentroid()[0]  # Graph Centroid
    # Search each cluster for a controller to add to the actions variable
    for index, cluster in enumerate(self.clusters):
        # if the centroid is in the cluster being iterated over, skip that cluster
        if self.graph.nodes[centroid]['cluster'] == index:
            continue

        bestNode = None
        lowestDistance = 100000000
        # search for node in cluster with smallest distance to centroid node
        for node in cluster:
            if nx.shortest_path_length(self.graph, centroid, node, weight='weight') < lowestDistance:
                lowestDistance = nx.shortest_path_length(self.graph, centroid, node, weight='weight')
                bestNode = node
        actions.append(bestNode)

    bestNode = None
    lowestDistance = 10000000
    # search for best node in the cluster where the centroid node lies within
    for node in self.clusters[self.graph.nodes[centroid]['cluster']]:
        if self.calculateDistance(actions + [node]) < lowestDistance:
            lowestDistance = self.calculateDistance(actions + [node])
            bestNode = node
    actions.append(bestNode)

    # Simulated Annealing Meta Heuristic
    current_state = actions
    annealing_rate = 0.90
    for x in range(1000):
        temperature = 1
        # choose a random already-selected node
        randomCluster = np.random.randint(len(self.clusters))
        randomBestAction = current_state[randomCluster]
        # get the neighbors of the randomly selected controller
        neighborList = [v for k, v in self.graph.edges if
                        (k == randomBestAction and v in self.clusters[self.graph.nodes[randomBestAction]['cluster']]) or
                        (v == randomBestAction and k in self.clusters[self.graph.nodes[randomBestAction]['cluster']])]
        neighborList.append(randomBestAction)
        # Get a random neighbor
        randomNeighbor = np.random.choice(neighborList)
        # create an alternate controller list with the neighbor replacing the previous controller
        proposed_state = current_state.copy()
        proposed_state[randomCluster] = randomNeighbor
        # get a random float between 0 and 1 to be the probability of changing the controller list
        threshold = float(np.random.rand(1))
        proposed_distance = self.calculateDistance(proposed_state)
        current_distance = self.calculateDistance(current_state)
        # if the new controller list is better than the original controller list, change to the new controller list
        # else change to the worst controller list with probability e^(-(new_distance - old_distance) / temperature)
        probability = 1 if proposed_distance < current_distance else np.exp(
            -(proposed_distance - current_distance) / temperature)
        if probability >= threshold:
            current_state = proposed_state.copy()
        temperature *= annealing_rate
    # ensure that the function returns the best of the original heuristic or the SA version
    if self.calculateDistance(actions) < self.calculateDistance(current_state):
        current_state = actions
    return current_state, self.calculateDistance(current_state)