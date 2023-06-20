import torch.nn as nn
import torch
from torchvision import datasets, transforms
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from datasets.PowerFlowData import PowerFlowData
from utils.custom_loss_functions import PowerImbalance

def main():
    # TODO import trainset, select an data.y, calculate the imbalance
    trainset = PowerFlowData(root='data', case='9', split=[.5, .3, .2], task='train',
                             normalize=False)
    sample = trainset[2]
    # loss_fn = PowerImbalance(trainset.xymean, trainset.xystd)
    loss_fn = PowerImbalance(torch.zeros(1), torch.ones(1))
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
    
    # loss = loss_fn(x, edge_index, edge_attr)
    sample.y = sample.y[:, 2:]
    loss = loss_fn(sample.y, sample.edge_index, sample.edge_attr)
    print(loss)
    
if __name__ == '__main__':
    main()