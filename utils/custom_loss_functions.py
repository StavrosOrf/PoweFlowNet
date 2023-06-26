import torch.nn as nn
import torch
from torchvision import datasets, transforms
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
import networkx


class Masked_L2_loss(nn.Module):
    """
    Custom loss function for the masked L2 loss.

    Args:
        output (torch.Tensor): The output of the neural network model.
        target (torch.Tensor): The target values.
        mask (torch.Tensor): The mask for the target values.

    Returns:
        torch.Tensor: The masked L2 loss.
    """

    def __init__(self):
        super(Masked_L2_loss, self).__init__()

    def forward(self, output, target, mask):

        mask = mask.type(torch.bool)

        # output = output * mask
        # target = target * mask
        output = torch.masked_select(output, mask)
        target = torch.masked_select(target, mask)

        criterion = nn.MSELoss(reduction='mean')
        loss = criterion(output, target)
        return loss

class PowerImbalance(MessagePassing):
    """Power Imbalance Loss Class

    Arguments:
        xymean: mean of the node features
        xy_std: standard deviation of the node features
        reduction: (str) 'sum' or 'mean' (node/batch-wise). P and Q are always added. 
        
    Input:
        x: node features        -- (N, 6)
        edge_index: edge index  -- (2, num_edges)
        edge_attr: edge features-- (num_edges, 2)
    """
    base_sn = 100 # kva
    base_voltage = 345 # kv
    base_ohm = 1190.25 # v**2/sn
    def __init__(self, xymean, xystd, edgemean, edgestd, reduction='mean'):
        super().__init__(aggr='add', flow='target_to_source')
        if xymean.shape[0] > 1:
            xymean = xymean[0:1]
        if xystd.shape[0] > 1:
            xystd = xystd[0:1]
        self.xymean = xymean
        self.xystd = xystd
        self.edgemean = edgemean
        self.edgestd = edgestd
        
    def de_normalize(self, x, edge_attr):
        return x * self.xystd + self.xymean, edge_attr * self.edgestd + self.edgemean
    
    def is_directed(self, edge_index):
        'determine if a graph id directed by reading only one edge'
        return edge_index[0,0] not in edge_index[1,edge_index[0,:] == edge_index[1,0]]
    
    def undirect_graph(self, edge_index, edge_attr):
        """transform a directed graph (index, attr) into undirect by duplicating and reversing the directed edges

        Arguments:
            edge_index -- shape (2, E)
            edge_attr -- shape (E, fe)
        """
        edge_index_dup = torch.stack(
            [edge_index[1,:], edge_index[0,:]],
            dim = 0
        )   # (2, E)
        edge_index = torch.cat(
            [edge_index, edge_index_dup],
            dim = 1
        )   # (2, 2*E)
        edge_attr = torch.cat(
            [edge_attr, edge_attr],
            dim = 0
        )   # (2*E, fe)
        
        return edge_index, edge_attr
    
    def message(self, x_i, x_j, edge_attr):
        """calculate injected power Pji
        
        Formula:
        $$
        P_{ji} = V_m^i*V_m^j*Y_{ij}*\cos(V_a^i-V_a^j-\theta_{ij})
                -(V_m^i)^2*Y_{ij}*\cos(-\theta_{ij})
        $$
        $$
        Q_{ji} = V_m^i*V_m^j*Y_{ij}*\sin(V_a^i-V_a^j-\theta_{ij})
                -(V_m^i)^2*Y_{ij}*\sin(-\theta_{ij})
        $$
        
        Input:
            x_i: (num_edges, 6)
            x_j: (num_edges, 6)
            edge_attr: (num_edges, 2)
        
        Return:
            Pji|Qji: (num_edges, 2)
        """
        r_x = edge_attr[:, 0:2] # (num_edges, 2)
        r, x = r_x[:, 0:1], r_x[:, 1:2]
        # zm_ij = torch.norm(r_x, p=2, dim=-1, keepdim=True) # (num_edges, 1) NOTE (r**2+x**2)**0.5 should be non-zero
        # za_ij = torch.acos(edge_attr[:, 0:1] / zm_ij) # (num_edges, 1)
        # ym_ij = 1/(zm_ij + 1e-6)        # (num_edges, 1)
        # ya_ij = -za_ij      # (num_edges, 1)    
        # g_ij = ym_ij * torch.cos(ya_ij) # (num_edges, 1)
        # b_ij = ym_ij * torch.sin(ya_ij) # (num_edges, 1)
        g_ij = r / (r**2 + x**2)
        b_ij = -x / (r**2 + x**2)
        ym_ij = torch.sqrt(g_ij**2+b_ij**2)
        ya_ij = torch.acos(g_ij/ym_ij)
        vm_i = x_i[:, 0:1] # (num_edges, 1)
        va_i = 1/180.*torch.pi*x_i[:, 1:2] # (num_edges, 1)
        vm_j = x_j[:, 0:1] # (num_edges, 1)
        va_j = 1/180.*torch.pi*x_j[:, 1:2] # (num_edges, 1)
        e_i = vm_i * torch.cos(va_i)
        f_i = vm_i * torch.sin(va_i)
        e_j = vm_j * torch.cos(va_j)
        f_j = vm_j * torch.sin(va_j)
        
        ####### my (incomplete) method #######
        # Pji = vm_i * vm_j * ym_ij * torch.cos(va_i - va_j - ya_ij) \
        #         - vm_i**2 * ym_ij * torch.cos(-ya_ij)
        # Qji = vm_i * vm_j * ym_ij * torch.sin(va_i - va_j - ya_ij) \
        #         - vm_i**2 * ym_ij * torch.sin(-ya_ij)
        
        ####### standard method #######
        # cannot be done since there's not complete information about whole neighborhood. 
        
        ####### another reference method #######
        # Pji = vm_i * vm_j * (g_ij*torch.cos(va_i-va_j)+b_ij*torch.sin(va_i-va_j))
        # Qji = vm_i * vm_j * (g_ij*torch.sin(va_i-va_j)-b_ij*torch.cos(va_i-va_j))
        
        ####### reference method 3 #######
        # Pji = g_ij*(vm_i**2 - vm_i*vm_j*torch.cos(va_i-va_j)) \
        #     - b_ij*(vm_i*vm_j*torch.sin(va_i-va_j))
        # Qji = b_ij*(- vm_i**2 + vm_i*vm_j*torch.cos(va_i-va_j)) \
        #     - g_ij*(vm_i*vm_j*torch.sin(va_i-va_j))
            
        ###### another mine ######
        Pji = g_ij*(e_i*e_j-e_i**2+f_i*f_j-f_i**2) + b_ij*(f_i*e_j-e_i*f_j)
        Qji = g_ij*(f_i*e_j-e_i*f_j) + b_ij*(-e_i*e_j+e_i**2-f_i*f_j+f_i**2)
        
        # --- DEBUG ---
        # self._dPQ = torch.cat([Pji, Qji], dim=-1) # (num_edges, 2)
        # --- DEBUG ---
        
        return torch.cat([Pji, Qji], dim=-1) # (num_edges, 2)
    
    def update(self, aggregated, x):
        """calculate power imbalance at each node

        Arguments:
            aggregated -- output of aggregation,    (num_nodes, 2)
            x -- node features                      (num_nodes, 6)
            
        Return:
            dPi|dQi: (num_nodes, 2)
        
        Formula:
        $$
            \Delta P_i = \sum_{j\in N_i} P_{ji} - P_{ij}
        $$
        """
        # TODO check if the aggregated result is correct
        
        # --- DEBUG ---
        # self.node_dPQ = self._is_i.float() @ self._dPQ # correct, gecontroleerd.
        # --- DEBUG ---
        dPi = - aggregated[:, 0:1] + x[:, 2:3] # (num_nodes, 1)
        dQi = - aggregated[:, 1:2] + x[:, 3:4] # (num_nodes, 1)

        return torch.cat([dPi, dQi], dim=-1) # (num_nodes, 2)
        
    def forward(self, x, edge_index, edge_attr):
        """calculate power imbalance at each node

        Arguments:
            x -- _description_
            edge_index -- _description_
            edge_attr -- _description_
        
        Return:
            dPQ: torch.float
        
        Formula:
        $$
            \Delta P_i = \sum_{j\in N_i} P_{ji} - P_{ij}
        $$
        """
        if self.is_directed(edge_index):
            edge_index, edge_attr = self.undirect_graph(edge_index, edge_attr)
        x, edge_attr = self.de_normalize(x, edge_attr)    # correct, gecontroleerd. 
        # --- per unit --- 
        # edge_attr[:, 0:2] = edge_attr[:, 0:2]/self.base_ohm
        # x[:, 2:4] = x[:, 2:4]/self.base_sn
        # --- DEBUG ---
        # self._edge_index = edge_index
        # self._is_i = torch.arange(14).view((14,1)).expand((14, 20)).long() == edge_index[0:1,:]
        # self._is_j = torch.arange(14).view((14,1)).expand((14, 20)).long() == edge_index[1:2,:]
        # --- DEBUG ---        
        dPQ = self.propagate(edge_index, x=x, edge_attr=edge_attr) # (num_nodes, 2)
        dPQ = dPQ.square().sum(dim=-1) # (num_nodes, 1)
        mean_dPQ = dPQ.mean()
        
        return mean_dPQ
    

def main():
    # TODO import trainset, select an data.y, calculate the imbalance
    # trainset = PowerFlowData(root='~/data/volume_2/power_flow_dataset', case='14', split=[.5, .3, .2], task='train')
    # sample = trainset[3]
    loss_fn = PowerImbalance(0, 1)
    x = torch.arange(18).reshape((3, 6)).float()
    edge_index = torch.tensor([
        [0, 1, 1, 2],
        [1, 0, 2, 1]
    ]).long()
    edge_attr = torch.tensor([
        [1, 0],
        [2, 0],
        [3, 0],
        [4, 0]
    ]).float()
    
    loss = loss_fn(x, edge_index, edge_attr)
    # loss = loss_fn(sample.y, sample.edge_index, sample.edge_attr)
    print(loss)
    
if __name__ == '__main__':
    main()