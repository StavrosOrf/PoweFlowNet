import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, TAGConv
from torch_geometric.utils import degree

class EdgeAggregation(MessagePassing):
    """MessagePassing for aggregating edge features

    """
    def __init__(self, nfeature_dim, efeature_dim, hidden_dim, output_dim):
        super().__init__(aggr='add')
        self.nfeature_dim = nfeature_dim
        self.efeature_dim = efeature_dim
        self.output_dim = output_dim

        # self.linear = nn.Linear(nfeature_dim, output_dim) 
        self.edge_aggr = nn.Sequential(
            nn.Linear(nfeature_dim*2 + efeature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def message(self, x_i, x_j, edge_attr):
        '''
        x_j:        shape (N, nfeature_dim,)
        edge_attr:  shape (N, efeature_dim,)
        '''
        return self.edge_aggr(torch.cat([x_i, x_j, edge_attr], dim=-1)) # PNAConv style
    
    def forward(self, x, edge_index, edge_attr):
        '''
        input:
            x:          shape (N, num_nodes, nfeature_dim,)
            edge_attr:  shape (N, num_edges, efeature_dim,)
            
        output:
            out:        shape (N, num_nodes, output_dim,)
        '''
        # Step 1: Add self-loops to the adjacency matrix.
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0)) # no self loop because NO EDGE ATTR FOR SELF LOOP
        
        # Step 2: Calculate the degree of each node.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col] 
        
        # Step 3: Feature transformation. 
        # x = self.linear(x) # no feature transformation
        
        # Step 4: Propagation
        out = self.propagate(x=x, edge_index=edge_index, edge_attr=edge_attr, norm=norm)
        #   no bias here
        
        return out

class MPN(nn.Module):
    """Wrapped Message Passing Network
        - One-time Message Passing to aggregate edge features into node features
        - Multiple Conv layers
    """
    def __init__(self, nfeature_dim, efeature_dim, output_dim, hidden_dim, n_gnn_layers, K, dropout_rate):
        super().__init__()
        self.nfeature_dim = nfeature_dim
        self.efeature_dim = efeature_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_gnn_layers = n_gnn_layers
        self.K = K
        self.dropout_rate = dropout_rate
        self.edge_aggr = EdgeAggregation(nfeature_dim, efeature_dim, hidden_dim, hidden_dim)
        self.convs = nn.ModuleList()

        if n_gnn_layers == 1:
            self.convs.append(TAGConv(hidden_dim, output_dim, K=K))
        else:
            self.convs.append(TAGConv(hidden_dim, hidden_dim, K=K))

        for l in range(n_gnn_layers-2):
            self.convs.append(TAGConv(hidden_dim, hidden_dim, K=K))
            
        self.convs.append(TAGConv(hidden_dim, output_dim, K=K))

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_features = data.edge_attr
        
        x = self.edge_aggr(x, edge_index, edge_features)
        for i in range(len(self.convs)-1):
            # x = self.convs[i](x=x, edge_index=edge_index, edge_weight=edge_attr)
            x = self.convs[i](x=x, edge_index=edge_index)
            x = nn.Dropout(self.dropout_rate, inplace=False)(x)
            x = nn.PReLU()(x)
        
        # x = self.convs[-1](x=x, edge_index=edge_index, edge_weight=edge_attr)
        x = self.convs[-1](x=x, edge_index=edge_index)
        
        return x