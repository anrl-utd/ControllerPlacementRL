import graph_nets as gn
import sonnet as snt
from controller_env.envs.graph_env import generateGraph
import networkx as nx

# Provide your own functions to generate graph-structured data.
graph, clusters, pos = generateGraph(9, 135, draw=False)
input_graphs = gn.utils_np.networkx_to_data_dict(graph, 1, 1)

# Create the graph network.
graph_net_module = gn.modules.GraphNetwork(
    edge_model_fn=lambda: snt.nets.MLP([32, 32]),
    node_model_fn=lambda: snt.nets.MLP([32, 32]),
    global_model_fn=lambda: snt.nets.MLP([32, 32]))

# Pass the input graphs to the graph network, and return the output graphs.
output_graphs = graph_net_module(input_graphs)