from typing import Callable, Optional, List, Tuple, Union

import torch
from torch_geometric.loader import DataLoader
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import torch.nn as nn

def train_epoch(
        model: nn.Module, 
        loader: DataLoader, 
        loss_fn: Callable, 
        optimizer: Optimizer, 
        device: torch.device
    ) -> float:
    """
    Trains a neural network model for one epoch using the specified data loader and optimizer.

    Args:
        model (nn.Module): The neural network model to be trained.
        loader (DataLoader): The PyTorch Geometric DataLoader containing the training data.
        optimizer (torch.optim.Optimizer): The PyTorch optimizer used for training the model.
        device (str): The device used for training the model (default: 'cpu').

    Returns:
        float: The mean loss value over all the batches in the DataLoader.

    """
    model = model.to(device)
    total_loss = 0.
    num_samples = 0
    model.train()
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = loss_fn(out, data.y)
        loss.backward()
        optimizer.step()
        num_samples += len(data)
        total_loss += loss.item() * len(data)
    
    mean_loss = total_loss / num_samples
    return mean_loss