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
    trainset = PowerFlowData(root='data', case='118mini', split=[.5, .3, .2], task='train',
                             normalize=True)
    valset = PowerFlowData(root='~/data/volume_2/power_flow_dataset', case='118v2', split=[.5, .3, .2], task='val', normalize=True)
    # sample = trainset[3]
    # # loss_fn = PowerImbalance(trainset.xymean, trainset.xystd)
    # loss_fn = PowerImbalance(*trainset.get_data_means_stds())
    # x = torch.arange(18).reshape((3, 6)).float()
    # edge_index = torch.tensor([
    #     [0, 1, 1, 2],
    #     [1, 0, 2, 1]
    # ]).long()
    # edge_attr = torch.tensor([
    #     [1, 0],
    #     [2, 0],
    #     [3, 0],
    #     [4, 0]
    # ]).float()
    
    # # loss = loss_fn(x, edge_index, edge_attr)
    # sample.y = sample.y[:, :]
    # loss = loss_fn(sample.y, sample.edge_index, sample.edge_attr)
    # print(loss)
    
    loss_fn = PowerImbalance(*valset.get_data_means_stds())
    val_loss = 0.
    for i in range(len(valset)):
        val_loss += loss_fn(valset[i].y, valset[i].edge_index, valset[i].edge_attr)
    
    val_loss /= len(valset)
    with open('val_loss.txt', 'w') as f:
        f.write(str(val_loss))
    
if __name__ == '__main__':
    main()