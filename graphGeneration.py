def generateAlternateGraph(num_clusters: int, num_nodes: int, weight_low: int = 0, weight_high: int = 100,
                           draw=True) -> (nx.Graph, list, dict):
    """
    Generates graph given number of clusters and nodes
    Args:
        num_clusters: Number of clusters
        num_nodes: Number of nodes
        weight_low: Lowest possible weight for edge in graph
        weight_high: Highest possible weight for edge in graph
        draw: Whether or not to show graph (True indicates to show)

    Returns:
        Graph with nodes in clusters, array of clusters, graph position for drawing
    """
    node_colors = np.arange(0, num_nodes, 1, np.uint8)  # Stores color of nodes
    nodes_per_cluster = int(num_nodes / num_clusters)
    clusters = np.zeros((num_clusters, nodes_per_cluster), np.uint8)  # Stores nodes in each cluster
    node_colors = node_colors // nodes_per_cluster  # Groups the node_colors
    G = nx.Graph()

    cluster_endpoints = []
    cluster = nx.full_rary_tree(int(np.log2(nodes_per_cluster)), nodes_per_cluster)
    temp = 0  # variable used to ensure diameter is as small as possible
    while nx.diameter(cluster) > (np.log2(nodes_per_cluster) + temp):
        cluster = nx.full_rary_tree(int(np.log2(nodes_per_cluster)), nodes_per_cluster)
        temp += 1
    nx.set_node_attributes(cluster, 1, 'cluster')
    clusters[0, :] = np.asarray(cluster.nodes)

    inner_cluster_edges = np.random.randint(0, nodes_per_cluster,
                                            (int(np.log2(nodes_per_cluster)), 2))
    cluster.add_edges_from(inner_cluster_edges)

    G = nx.disjoint_union(G, cluster)
    for i in range(1, num_clusters):
        cluster = nx.full_rary_tree(int(np.log2(nodes_per_cluster)), nodes_per_cluster)
        temp = 0
        while nx.diameter(cluster) > (np.log2(nodes_per_cluster) + temp):
            cluster = nx.full_rary_tree(int(np.log2(nodes_per_cluster)), nodes_per_cluster)
            temp += 1

        nx.set_node_attributes(cluster, 1, 'cluster')
        clusters[i, :] = np.asarray(cluster.nodes) + nodes_per_cluster * i
        G = nx.disjoint_union(G, cluster)
        cluster_endpoint = np.random.randint(0, nodes_per_cluster)
        cluster_endpoints.append(cluster_endpoint)
        G.add_edge(cluster_endpoint, np.random.randint(nodes_per_cluster * i, nodes_per_cluster * i + i))

    # adding inter and inner edges of the clusters
    closest_length = 1000
    nearest_cluster = 0
    shortest_path = 0
    for i in range(1, num_clusters):
        # check for closest cluster besides main cluster
        for x in range(2, num_clusters - 1):
            shortest_path = nx.shortest_path_length(G, cluster_endpoints[i - 1], cluster_endpoints[x - 1])
            if shortest_path < closest_length:
                closest_length = shortest_path
                nearest_cluster = x

        # add inner_cluster_edges
        inner_cluster_edges = np.random.randint(nodes_per_cluster * i, nodes_per_cluster * i + nodes_per_cluster,
                                                (int(np.log2(nodes_per_cluster)), 2))
        G.add_edges_from(inner_cluster_edges)

        # if the nearest_cluster is too far away, don't add inter-cluster edges
        if shortest_path > (np.random.randint(np.log2(nodes_per_cluster), np.log2(nodes_per_cluster) + 1)):
            continue

        # add inter_cluster_edges
        inter_cluster_edges = np.random.randint(nodes_per_cluster * i, nodes_per_cluster * i + nodes_per_cluster,
                                                (int(nodes_per_cluster / (
                                                            np.random.randint(0, (np.log2(nodes_per_cluster))) + 1))))
        inter_cluster_edges = [[y, np.random.randint(nodes_per_cluster * nearest_cluster,
                                                     nodes_per_cluster * nearest_cluster + nodes_per_cluster)] for y in
                               inter_cluster_edges]

        G.add_edges_from(inter_cluster_edges)
    G.remove_edges_from(nx.selfloop_edges(G))  # Remove self-loops caused by adding random edge

    # Add edge weights to the whole graph
    for (u, v) in G.edges():
        G.edges[u, v]['weight'] = np.random.random() * (weight_high - weight_low) + weight_low
    pos = nx.spring_layout(G)

    # Draw graph
    if draw:
        nx.draw_networkx_nodes(G, pos, node_color=node_colors)
        nx.draw_networkx_labels(G, pos)
        nx.draw_networkx_edges(G, pos, G.edges())
        plt.draw()
        plt.show()

    return G, clusters, pos