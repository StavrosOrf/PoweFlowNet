import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self, nfeature_dim, output_dim, hidden_dim, **kwargs):
        super(GCN, self).__init__()        
        self.conv1 = GCNConv(nfeature_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index 
        # x = x[:, 4:10]        
               
        x = self.relu(self.conv1(x, edge_index))
        x = self.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)

        return x