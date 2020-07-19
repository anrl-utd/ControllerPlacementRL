def add_access_nodes(topology_file, num_trees, weight_low = 1, weight_high = 10, tree_height= 0, tree_branching_factor = 0):
    G = nx.read_gml(topology_file, label="id")
    G = nx.Graph(G)

    original_nodes = G.nodes

    # find thhe nodes with least amount of neighbors
    nodes_neighbors = {}
    for x in range(0, len(G.nodes)):
        nodes_neighbors[x] = len(G.edges(x))
    sorted_nodes_neighbors = {k: v for k, v in sorted(nodes_neighbors.items(), key=lambda item: item[1])}
    x = 0
    while(x < num_trees):
        starting_tree_node = len(G.nodes)
        if np.random.random() < 0.25:
            x = x + 1
            num_trees = num_trees + 1
        if tree_branching_factor == 0 and tree_height == 0:
            tree = nx.balanced_tree(np.random.randint(2, 4), np.random.randint(1, 3))
        elif tree_height != 0 and tree_branching_factor != 0:
            tree = nx.balanced_tree(tree_branching_factor, tree_height)
        elif tree_height != 0:
            tree = nx.balacned_tree(np.random.randint(2, 4), tree_height)
        else:
            tree = nx.balanced_tree(tree_branching_factor, np.random.randint(2,3))
        G = nx.disjoint_union(G, tree)
        G.add_edge(x, starting_tree_node, weight = np.random.randint(0, (weight_high - weight_low)) + weight_low)
        x = x + 1


    pos = nx.spring_layout(G)
    node_colors = np.ones(len(G.nodes))* 2
    node_colors[original_nodes] = np.ones(len(original_nodes))
    # Draw graph
    nx.draw_networkx_nodes(G, pos, node_color=node_colors)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, G.edges())
    plt.draw()
    plt.show()