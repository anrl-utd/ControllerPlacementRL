from setuptools import setup
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import warnings
import random
warnings.filterwarnings("ignore", category=UserWarning)

while True:
    numClusters = int(input("How many clusters do you want? "))

    n = int(input("How many nodes do you want? "))
    p = .2

    graph = nx.powerlaw_cluster_graph(n,3, p, random.seed(a = None, version = 2))

    traversal = list(nx.bfs_tree(graph, source = 0))    # get a bft of the graph in a list
    array_traversal = np.array_split(traversal, numClusters)      # split the bft list into equal parts

    pos = nx.spring_layout(graph)   # get the positions of the nodes of the graph
    for index, the_traversal in enumerate(array_traversal): # for each cluster and index of the graph
        subpos = dict()                                # create a sub dictionary of positions for each cluster's ndes
        for key, value in pos.items():
            if key in the_traversal:
                subpos[key] = value
        randColor = "#" + str(random.randint(0, 999999)).zfill(6)   # get a random hexadecimal colour
        nx.draw_networkx_nodes(graph.subgraph(the_traversal), subpos, node_color=randColor, label = "Cluster " + str(index))  # draw an individual cluster
    nx.draw_networkx_edges(graph, pos, graph.edges())        # draw the edges of the graph
    nx.draw_networkx_labels(graph, pos)                      # draw the labels of the graph

    plt.savefig("path.png")
    plt.draw()
    plt.legend()
    plt.show()



#
# setup(name='controller_env',
#       version='0.0.1',
#       install_requires=['gym']#And any other dependencies required
# )


