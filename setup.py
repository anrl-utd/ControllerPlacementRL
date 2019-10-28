from setuptools import setup
import networkx as nx
import matplotlib.pyplot as plt
import warnings
import random
warnings.filterwarnings("ignore", category=UserWarning)

while True:
    n = int(input())
    p = .2

    graph = nx.powerlaw_cluster_graph(n,3, p, random.seed(a = None, version = 2))

    nx.draw(graph, with_labels=True)
    plt.savefig("path.png")
    plt.draw()
    plt.show()


#
# setup(name='controller_env',
#       version='0.0.1',
#       install_requires=['gym']#And any other dependencies required
# )


