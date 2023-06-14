import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, TAGConv, GCNConv, GATv2Conv
from torch_geometric.utils import degree
from torch_geometric.nn.pool import MemPooling

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
        deg_inv_sqrt = torch.rsqrt(deg)
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
        self.edge_aggr2 = EdgeAggregation(hidden_dim, efeature_dim, hidden_dim, hidden_dim)
        self.convs = nn.ModuleList()

        self.cluster1 = MemPooling(16, hidden_dim, 1, 3)


        if n_gnn_layers == 1:
            self.convs.append(TAGConv(hidden_dim, output_dim, K=K))
        else:
            self.convs.append(TAGConv(hidden_dim, hidden_dim, K=K))

        for l in range(n_gnn_layers-2):
            self.convs.append(TAGConv(hidden_dim, hidden_dim, K=K))
            
        self.convs.append(TAGConv(hidden_dim, output_dim, K=K))

    def forward(self, data):
        assert data.x.shape[-1] == self.nfeature_dim * 2 + 4 # features and their mask + one-hot node type embedding
        x = data.x[:, 4:4+self.nfeature_dim] # first four features: node type. not elegant at all this way. just saying. 
        input_x = x # problem if there is inplace operation on x, so pay attention
        mask = data.x[:, -self.nfeature_dim:]# last few dimensions: mask.
        edge_index = data.edge_index
        edge_features = data.edge_attr

        # print("=SRT=======>", data.x.shape)
        # output_x = self.cluster1(data.x, data.batch)
        # print("=END=======>", output_x[0].shape, output_x[1].shape)
        # print("=END=======>", output_x[0])
        # print("=END=======>", output_x[1])
        print("====", data.edge_attr.shape)
        
        x = self.edge_aggr(x, edge_index, edge_features)
        for i in range(len(self.convs)-1):
            # x = self.convs[i](x=x, edge_index=edge_index, edge_weight=edge_attr)
            x = self.convs[i](x=x, edge_index=edge_index)
            x = nn.Dropout(self.dropout_rate, inplace=False)(x)
            x = nn.ReLU()(x)
            x = self.edge_aggr2(x, edge_index, edge_features)
        
        # x = self.convs[-1](x=x, edge_index=edge_index, edge_weight=edge_attr)
        x = self.convs[-1](x=x, edge_index=edge_index)
        
        return x
    
# class HierarchicalClusterGCN(nn.Module):
#     """Wrapped Message Passing Network
#         - One-time Message Passing to aggregate edge features into node features
#         - Multiple Conv layers
#     """
#     def __init__(self, nfeature_dim, efeature_dim, output_dim, hidden_dim, n_gnn_layers, K, dropout_rate):
#         super().__init__()
#         self.nfeature_dim = nfeature_dim
#         self.efeature_dim = efeature_dim
#         self.output_dim = output_dim
#         self.hidden_dim = hidden_dim
#         self.n_gnn_layers = n_gnn_layers
#         self.K = K
#         self.dropout_rate = dropout_rate

#         self.grounds_convs = nn.ModuleList()
#         self.edge_aggr_grounds = EdgeAggregation(nfeature_dim, efeature_dim, hidden_dim, hidden_dim)
#         self.grounds_convs.append(TAGConv(hidden_dim, hidden_dim, K=K))
#         for l in range(n_gnn_layers-2):
#             self.grounds_convs.append(TAGConv(hidden_dim, hidden_dim, K=K))

#         self.cluster1 = MemPooling()
#         self.cluster1_convs = nn.ModuleList()
#         self.edge_aggr_cluster1 = EdgeAggregation(nfeature_dim, efeature_dim, hidden_dim, hidden_dim)
#         self.cluster1_convs.append(TAGConv(hidden_dim, hidden_dim, K=K))
#         for l in range(n_gnn_layers-2):
#             self.convs.append(TAGConv(hidden_dim, hidden_dim, K=K))

#         self.cluster2 = MemPooling()
#         self.cluster2_convs = nn.ModuleList()
#         self.edge_aggr_cluster2 = EdgeAggregation(nfeature_dim, efeature_dim, hidden_dim, hidden_dim)
#         self.cluster2_convs.append(TAGConv(hidden_dim, hidden_dim, K=K))
#         for l in range(n_gnn_layers-2):
#             self.convs.append(TAGConv(hidden_dim, hidden_dim, K=K))

#     def forward(self, data):
#         x = data.x
#         edge_index = data.edge_index
#         edge_features = data.edge_attr
        
#         xg = self.edge_aggr(x, edge_index, edge_features)
#         for i in range(len(self.convs)-1):
#             xg = self.convs[i](x=x, edge_index=edge_index)
#             xg = nn.Dropout(self.dropout_rate, inplace=False)(x)
#             xg = nn.ReLU()(x)
#         xg = self.convs[-1](x=x, edge_index=edge_index)

#         x1 = self.cluster1()
#         x1 = self.edge_aggr(x, edge_index, edge_features)
#         for i in range(len(self.convs)-1):
#             x1 = self.convs[i](x=x, edge_index=edge_index)
#             x1 = nn.Dropout(self.dropout_rate, inplace=False)(x)
#             x1 = nn.ReLU()(x)
#         x1 = self.convs[-1](x=x, edge_index=edge_index)


#         x2 = self.edge_aggr(x, edge_index, edge_features)
#         for i in range(len(self.convs)-1):
#             x2 = self.convs[i](x=x, edge_index=edge_index)
#             x2 = nn.Dropout(self.dropout_rate, inplace=False)(x)
#             x2 = nn.ReLU()(x)
#         x2 = self.convs[-1](x=x, edge_index=edge_index)

        
#         return x


class SimpleNetwork(nn.Module):
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
        self.convs = nn.ModuleList()

        self.edge_aggr = nn.Sequential(
            nn.Linear(efeature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.convs.append(GCNConv(nfeature_dim, hidden_dim, K=K))

        for l in range(n_gnn_layers-2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim, K=K))
            
        self.convs.append(GCNConv(hidden_dim, output_dim, K=K))

    def forward(self, data):
        # assert data.x.shape[-1] == self.nfeature_dim * 2 + 4 # features and their mask + one-hot node type embedding
        x = data.x
        edge_index = data.edge_index
        edge_features = data.edge_attr
        edge_weights = self.edge_aggr(edge_features)
        
        for i in range(len(self.convs)-1):
            x = self.convs[i](x=x, edge_index=edge_index, edge_weight=edge_weights)
            # x = nn.Dropout(self.dropout_rate, inplace=False)(x)
            x = nn.ReLU()(x)
        
        x = self.convs[-1](x=x, edge_index=edge_index, edge_weight=edge_weights)
        
        return x
    
class SimpleNetwork2(nn.Module):
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
        self.convs = nn.ModuleList()

        self.edge_aggr = nn.Sequential(
            nn.Linear(efeature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.convs.append(GATv2Conv(nfeature_dim, hidden_dim // K, heads=K, edge_dim=efeature_dim))

        for l in range(n_gnn_layers-2):
            self.convs.append(GATv2Conv(hidden_dim, hidden_dim // K, heads=K, edge_dim=efeature_dim))
            
        self.convs.append(GATv2Conv(hidden_dim, output_dim, heads=1, edge_dim=efeature_dim))

    def forward(self, data):
        # assert data.x.shape[-1] == self.nfeature_dim * 2 + 4 # features and their mask + one-hot node type embedding
        x = data.x
        edge_index = data.edge_index
        edge_features = data.edge_attr
        
        for i in range(len(self.convs)-1):
            x = self.convs[i](x=x, edge_index=edge_index, edge_attr=edge_features)
            x = nn.ReLU()(x)
        
        x = self.convs[-1](x=x, edge_index=edge_index, edge_attr=edge_features)
        
        return x