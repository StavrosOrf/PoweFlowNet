import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

class GaussSeidelPooling(MessagePassing):
    def __init__(self, in_channels, out_channels, num_layers):
        super(GaussSeidelPooling, self).__init__(aggr='add')

        self.num_layers = num_layers
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        for _ in range(self.num_layers):
            x = self.propagate(edge_index, x=x)
        return x

    def message(self, x_j):
        return self.lin(x_j)

class GraphTransformerWithPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_rate, num_pool_layers):
        super(GraphTransformerWithPooling, self).__init__()

        self.num_pool_layers = num_pool_layers

        self.conv1 = GaussSeidelPooling(input_dim, hidden_dim, num_pool_layers)
        self.conv2 = GaussSeidelPooling(hidden_dim, hidden_dim, num_pool_layers)
        self.lin = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.lin(x)
        return x
