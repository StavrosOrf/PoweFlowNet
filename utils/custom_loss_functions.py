import torch.nn as nn
import torch
from torchvision import datasets, transforms
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


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

    def __init__(self, regularize=True, regcoeff=1):
        super(Masked_L2_loss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.regularize = regularize
        self.regcoeff = regcoeff

    def forward(self, output, target, mask):

        masked = mask.type(torch.bool)

        # output = output * mask
        # target = target * mask
        outputl = torch.masked_select(output, masked)
        targetl = torch.masked_select(target, masked)

        loss = self.criterion(outputl, targetl)

        if self.regularize:
            masked = (1 - mask).type(torch.bool)
            output_reg = torch.masked_select(output, masked)
            target_reg = torch.masked_select(target, masked)
            loss = loss + self.regcoeff * self.criterion(output_reg, target_reg)

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
    def __init__(self, xymean, xystd, reduction='mean'):
        super().__init__(aggr='add', flow='target_to_source')
        if xymean.shape[0] > 1:
            xymean = xymean[0:1]
        if xystd.shape[0] > 1:
            xystd = xystd[0:1]
        self.xymean = xymean
        self.xystd = xystd
        
    def de_normalize(self, x):
        return x * self.xystd + self.xymean
    
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
        zm_ij = torch.norm(r_x, p=2, dim=-1, keepdim=True) # (num_edges, 1) NOTE (r**2+x**2)**0.5 should be non-zero
        za_ij = torch.acos(edge_attr[:, 0:1] / zm_ij) # (num_edges, 1)
        ym_ij = 1/(zm_ij + 1e-6)        # (num_edges, 1)
        ya_ij = -za_ij      # (num_edges, 1)    
        vm_i = x_i[:, 0:1] # (num_edges, 1)
        va_i = x_i[:, 1:2] # (num_edges, 1)
        vm_j = x_j[:, 0:1] # (num_edges, 1)
        va_j = x_j[:, 1:2] # (num_edges, 1)
        
        ####### my (incomplete) method #######
        # Pji = vm_i * vm_j * ym_ij * torch.cos(va_i - va_j - ya_ij) \
        #         - vm_i**2 * ym_ij * torch.cos(-ya_ij)
        # Qji = vm_i * vm_j * ym_ij * torch.sin(va_i - va_j - ya_ij) \
        #         - vm_i**2 * ym_ij * torch.sin(-ya_ij)
        
        ####### standard method #######
        # cannot be done since there's not complete information about whole neighborhood. 
        
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
        dPi = aggregated[:, 0:1] - x[:, 2:3] # (num_nodes, 1)
        dQi = aggregated[:, 1:2] - x[:, 3:4] # (num_nodes, 1)

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
        x = self.de_normalize(x)    # correct, gecontroleerd. 
        dPQ = self.propagate(edge_index, x=x, edge_attr=edge_attr) # (num_nodes, 2)
        dPQ = dPQ.sum(dim=-1) # (num_nodes, 1)
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