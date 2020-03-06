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
        if index not in controllers:
            node_size[index] = 300
        else:
            node_size[index] = 1000
    edge_color = list(range(len(self.graph.edges)))
    shortest_paths = []
    for controller in controllers:
        shortest_path = nx.shortest_path(self.graph, controller, graph_centroid[0], weight='weight')
        shortest_paths = shortest_paths + [
            (min(shortest_path[index], shortest_path[index + 1]), max(shortest_path[index], shortest_path[index + 1]))
            for index in range(len(shortest_path) - 1)]
    for index in range(len(edge_color)):
        edge_color[index] = 1
    for index in range(len(shortest_paths)):
        edge_color[list(self.graph.edges()).index(shortest_paths[index])] = 2

    nx.draw_networkx_nodes(self.graph, self.pos, node_color=node_colors, node_size=node_size)
    nx.draw_networkx_edges(self.graph, self.pos, self.graph.edges(),
                           edge_color=edge_color)  # draw the edges of the self.graph
    nx.draw_networkx_labels(self.graph, self.pos)  # draw  the labels of the self.graph
    edge_labels = nx.get_edge_attributes(self.graph, 'weight')
    nx.draw_networkx_edge_labels(self.graph, self.pos,
                                 edge_labels=edge_labels)  # draw the edge weights of the self.graph
    plt.draw()
    plt.show()

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

def graphCentroidAction(self):
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
    return actions