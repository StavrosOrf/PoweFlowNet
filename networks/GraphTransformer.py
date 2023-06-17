import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv

class GraphTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_rate):
        super(GraphTransformer, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.relus = nn.ModuleList()

        self.convs.append(TransformerConv(input_dim, hidden_dim))
        self.relus.append(nn.ReLU())

        for _ in range(num_layers - 2):
            self.convs.append(TransformerConv(hidden_dim, hidden_dim))
            self.relus.append(nn.ReLU())

        self.convs.append(TransformerConv(hidden_dim, output_dim))
        self.dropout = nn.Dropout(dropout_rate)
        self.dropout_rate = dropout_rate  # Add dropout_rate attribute

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for layer in range(len(self.convs) - 1):
            x = self.convs[layer](x, edge_index)
            x = self.relus[layer](x)
            x = self.dropout(x)  # Remove p parameter

        x = self.convs[-1](x, edge_index)

        return x
