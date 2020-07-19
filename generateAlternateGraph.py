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
    total_nodes = 0
    temp_clusters = {}  # Temp storage for clusters
    remainder = num_nodes % num_clusters
    clusters = []  # Stores nodes in each cluster
    # organize number of nodes per cluster and assign node colors
    temp = 0
    # fill in cluster and temp cluster variables and set up node_colors variable
    for x in range(num_clusters):
        if remainder > x:
            nodes_per_cluster = int(num_nodes / num_clusters) + 1
        else:
            nodes_per_cluster = int(num_nodes / num_clusters)

        temp_clusters[x] = nodes_per_cluster
        node_colors[temp + np.arange(nodes_per_cluster)] = x
        temp += nodes_per_cluster
        clusters.append(list(np.arange(nodes_per_cluster) + total_nodes))
        total_nodes += nodes_per_cluster
    G = nx.Graph()

    cluster_endpoints = []

    # create first cluster
    cluster = nx.full_rary_tree(int(np.log2(temp_clusters[0])), temp_clusters[0])

    temp = 0  # variable used to ensure diameter is as small as possible
    while nx.diameter(cluster) > (np.log2(temp_clusters[0]) + temp):
        cluster = nx.full_rary_tree(int(np.log2(temp_clusters[0])),temp_clusters[0])
        temp += 1
    nx.set_node_attributes(cluster, 0, 'cluster')

    # set initial edge weight of first cluster
    for (u, v) in cluster.edges():
        cluster.edges[u, v]['weight'] = np.random.random() * 0.75 * (weight_high - weight_low) + weight_low * 1.25

    inner_cluster_edges = np.random.randint(0, temp_clusters[0],
                                            (int(np.log2(temp_clusters[0])), 2))

    # add edge weights to new edges of first cluster
    inner_cluster_edges =  [(u, v,  np.random.random() * 0.75 * (weight_high - weight_low) + weight_low * 1.25) for u,v in inner_cluster_edges]
    cluster.add_weighted_edges_from(inner_cluster_edges)

    G = nx.disjoint_union(G, cluster)

    # create other clusters
    for i in range(1, num_clusters):
        # create cluster
        cluster = nx.full_rary_tree(int(np.log2(temp_clusters[i])), temp_clusters[i])
        temp = 0
        while nx.diameter(cluster) > (np.log2(temp_clusters[i]) + temp):
            cluster = nx.full_rary_tree(int(np.log2(temp_clusters[i])), temp_clusters[i])
            temp += 1

        nx.set_node_attributes(cluster, i, 'cluster')

        # set initial edge weights
        for (u, v) in cluster.edges():
            if not(u in np.arange(temp_clusters[x]/2) or v in np.arange(temp_clusters[x] / 2)):
                cluster.edges[u, v]['weight'] = np.random.random() * 0.20 * (weight_high - weight_low) + weight_low * 1.05
            else:
                cluster.edges[u, v]['weight'] = np.random.random() * 0.05 * (weight_high - weight_low) + weight_low


        G = nx.disjoint_union(G, cluster)

        # add connections from new clusters to first cluster
        cluster_endpoint = np.random.randint(0, temp_clusters[i])
        cluster_endpoints.append(cluster_endpoint)
        G.add_edge(cluster_endpoint, np.random.randint(temp_clusters[i] * i, temp_clusters[i] * i + i), weight = np.random.random() * 0.20 * (weight_high - weight_low) + weight_low * 1.05)

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
        inner_cluster_edges = np.random.randint(temp_clusters[i] * i, temp_clusters[i] * i + temp_clusters[i],
                                                (int(np.log2(temp_clusters[i])), 2))
        inner_cluster_edges = [(u, v, np.random.random() * 0.05 * (weight_high - weight_low) + weight_low) for
                               u, v in inner_cluster_edges]
        # cluster.add_weighted_edges_from(inner_cluster_edges)
        G.add_weighted_edges_from(inner_cluster_edges)

        # if the nearest_cluster is too far away, don't add inter-cluster edges
        if shortest_path > (np.random.randint(np.log2(temp_clusters[i]), np.log2(temp_clusters[i]) + 1)):
            continue

        # add inter_cluster_edges
        inter_cluster_edges = np.random.randint(temp_clusters[i] * i, temp_clusters[i] * i + temp_clusters[i],
                                                (int(temp_clusters[i] / (
                                                            np.random.randint(0, (np.log2(temp_clusters[i]))) + 1))))
        inter_cluster_edges = [[y, np.random.randint(temp_clusters[i] * nearest_cluster,
                                                     temp_clusters[i] * nearest_cluster + temp_clusters[i]), np.random.random() * 0.20 * (weight_high - weight_low) + weight_low * 1.05] for y in
                               inter_cluster_edges]

        # cluster.add_weighted_edges_from(inner_cluster_edges)
        G.add_weighted_edges_from(inter_cluster_edges)
    G.remove_edges_from(nx.selfloop_edges(G))  # Remove self-loops caused by adding random edge

    pos = nx.spring_layout(G)

    # Draw graph
    if draw:
        nx.draw_networkx_nodes(G, pos, node_color=node_colors)
        nx.draw_networkx_labels(G, pos)
        # nx.draw_networkx_edge_labels(G, pos)
        nx.draw_networkx_edges(G, pos, G.edges())
        plt.draw()
        plt.show()

    return G, clusters, pos