import torch.nn as nn
import torch
from torchvision import datasets, transforms
from torch import nn
import torch.nn.functional as F


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
        #print("Output shape: ", output.shape)
        #print("Target shape: ", target.shape)
        #print("Mask shape: ", mask.shape) 
        output = torch.masked_select(output, mask)
        target = torch.masked_select(target, mask)

        criterion = nn.MSELoss(reduction='mean')
        loss = criterion(output, target)
        return loss
