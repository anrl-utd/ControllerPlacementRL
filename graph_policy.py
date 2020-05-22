import os.path as osp

import argparse
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GAE, VGAE
from torch_geometric.utils import train_test_split_edges
from torch.utils.data import IterableDataset
from torch_geometric.data import DataLoader, Data
from controller_env.envs.graph_env import generateGraph
from torch_geometric.utils.convert import from_networkx, from_scipy_sparse_matrix
import networkx as nx

torch.manual_seed(12345)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='GAE')
args = parser.parse_args()
assert args.model in ['GAE', 'VGAE']
kwargs = {'GAE': GAE, 'VGAE': VGAE}


class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)
        if args.model in ['GAE']:
            self.conv2 = GCNConv(2 * out_channels, out_channels, cached=True)
        elif args.model in ['VGAE']:
            self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=True)
            self.conv_logvar = GCNConv(2 * out_channels, out_channels,
                                       cached=True)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        if args.model in ['GAE']:
            return self.conv2(x, edge_index)
        elif args.model in ['VGAE']:
            return self.conv_mu(x, edge_index), self.conv_logvar(x, edge_index)

class NetworkIterable(IterableDataset):
	def __iter__(self):
		graph, cluster, pos = generateGraph(3, 15)
		print(graph.edges)
		print(graph.edges.data('weight'))
		#edges = nx.to_scipy_sparse_matrix(graph)
		#pytorch_edges = from_scipy_sparse_matrix(edges)
		data = from_networkx(graph)
		print(data)
		#data = Data(x=pytorch_graph, edge_index=pytorch_edges, y=pytorch_graph)
		data.train_mask = data.val_mask = data.test_mask = data.y = None
		data.weight = data.weight.long()
		data = train_test_split_edges(data)
		yield data

channels = 1  # Number of hidden channels
num_features = 1
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = kwargs[args.model](Encoder(num_features, channels)).to(dev)
graph_dataset = NetworkIterable()  # Generates graphs for the dataloader "on the fly"
loader = DataLoader(graph_dataset, batch_size=4)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train():
	for batch_data in loader:
		model.train()
		optimizer.zero_grad()
		print(batch_data)
		x, train_pos_edge_index = batch_data.batch.to(dev), batch_data.train_pos_edge_index.to(dev)
		print(x)
		print(train_pos_edge_index)
		z = model.encode(x, train_pos_edge_index)
		loss = model.recon_loss(z, train_pos_edge_index)
		if args.model in ['VGAE']:
			loss = loss + (1 / data.num_nodes) * model.kl_loss()
		loss.backward()
		optimizer.step()


def test(pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model.encode(x, train_pos_edge_index)
    return model.test(z, pos_edge_index, neg_edge_index)


for epoch in range(1, 401):
    train()
    auc, ap = test(data.test_pos_edge_index, data.test_neg_edge_index)
    print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))