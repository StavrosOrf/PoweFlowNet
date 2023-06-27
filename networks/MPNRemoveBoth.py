import torch
import torch.nn as nn
from torch_geometric.nn import TAGConv

class MPN(nn.Module):
    def __init__(self, nfeature_dim, output_dim, hidden_dim, n_gnn_layers, K, dropout_rate):
        super().__init__()
        self.nfeature_dim = nfeature_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_gnn_layers = n_gnn_layers
        self.K = K
        self.dropout_rate = dropout_rate
        self.convs = nn.ModuleList()

        #if n_gnn_layers == 1:
        #    self.convs.append(TAGConv(nfeature_dim, output_dim, K=K))
        #else:
        #    self.convs.append(TAGConv(nfeature_dim, hidden_dim, K=K))

        #for _ in range(n_gnn_layers-2):
        #    self.convs.append(TAGConv(hidden_dim, hidden_dim, K=K))

        self.convs.append(TAGConv(nfeature_dim, output_dim, K=K))

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index

        for i in range(len(self.convs)-1):
            x = self.convs[i](x=x, edge_index=edge_index)
            x = nn.Dropout(self.dropout_rate, inplace=False)(x)
            x = nn.ReLU()(x)

        x = self.convs[-1](x=x, edge_index=edge_index)

        return x
    
class MPN_simplenet(nn.Module):
    def __init__(self, nfeature_dim, output_dim, hidden_dim, n_gnn_layers, K, dropout_rate):
        super().__init__()
        self.nfeature_dim = nfeature_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_gnn_layers = n_gnn_layers
        self.K = K
        self.dropout_rate = dropout_rate
        self.convs = nn.ModuleList()

        #if n_gnn_layers == 1:
         #   self.convs.append(TAGConv(nfeature_dim, output_dim, K=K))
        #else:
        #    self.convs.append(TAGConv(nfeature_dim, hidden_dim, K=K))

        #for _ in range(n_gnn_layers-2):
        #    self.convs.append(TAGConv(hidden_dim, hidden_dim, K=K))

        self.convs.append(TAGConv(nfeature_dim, output_dim, K=K))

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index

        for i in range(len(self.convs)-1):
            x = self.convs[i](x=x, edge_index=edge_index)
            x = nn.Dropout(self.dropout_rate, inplace=False)(x)
            x = nn.ReLU()(x)

        x = self.convs[-1](x=x, edge_index=edge_index)

        return x

