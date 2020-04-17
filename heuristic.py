def best_action(self):
    actions = []
    centroid = self.findGraphCentroid()[0]
    for index, cluster in enumerate(self.clusters):
        if self.graph.nodes[centroid]['cluster'] == index:
            continue
        bestNode = None
        lowestDistance = 100000000
        for node in cluster:
            if nx.shortest_path_length(self.graph, centroid, node, weight='weight') < lowestDistance:
                lowestDistance = nx.shortest_path_length(self.graph, centroid, node, weight='weight')
                bestNode = node
        actions.append(bestNode)
    bestNode = None
    lowestDistance = 10000000
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
        randomCluster = np.random.randint(len(self.clusters))
        randomBestAction = current_state[randomCluster]
        neighborList = [v for k, v in self.graph.edges if
                        k == randomBestAction and v in self.clusters[self.graph.nodes[randomBestAction]['cluster']]]
        neighborList.append(randomBestAction)
        randomNeighbor = np.random.choice(neighborList)
        proposed_state = current_state.copy()
        proposed_state[randomCluster] = randomNeighbor
        threshold = float(np.random.rand(1))
        proposed_distance = self.calculateDistance(proposed_state)
        current_distance = self.calculateDistance(current_state)
        probability = 1 if proposed_distance < current_distance else np.exp(
            -(proposed_distance - current_distance) / temperature)
        if probability >= threshold:
            current_state = proposed_state.copy()
        temperature *= annealing_rate
    if self.calculateDistance(actions) < self.calculateDistance(current_state):
        current_state = actions
    return current_state, self.calculateDistance(current_state)