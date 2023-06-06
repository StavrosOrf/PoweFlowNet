"""This module provides functions for 
    - evaluation_epoch - evaluate performance over a whole epoch
    - other evaluation metrics function [NotImplemented]
"""
from typing import Callable

import torch
from torch_geometric.loader import DataLoader
from torch.optim.optimizer import Optimizer
import torch.nn as nn

def num_params(model: nn.Module) -> int:
    """
    Returns the number of trainable parameters in a neural network model.

    Args:
        model (nn.Module): The neural network model.

    Returns:
        int: The number of trainable parameters in the model.

    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@torch.no_grad()
def evaluate_epoch(
        model: nn.Module, 
        loader: DataLoader, 
        loss_fn: Callable,
        device: str='cpu') -> float:
    """
    Evaluates the performance of a trained neural network model on a dataset using the specified data loader.

    Args:
        model (nn.Module): The trained neural network model to be evaluated.
        loader (DataLoader): The PyTorch Geometric DataLoader containing the evaluation data.
        device (str): The device used for evaluating the model (default: 'cpu').

    Returns:
        float: The mean loss value over all the batches in the DataLoader.

    """
    model.eval()
    total_loss = 0.
    num_samples = 0
    for data in loader:
        data = data.to(device)
        out = model(data)
        loss = loss_fn(out, data.y)
        num_samples += len(data)
        total_loss += loss.item() * len(data)
        
    mean_loss = total_loss / num_samples
    return mean_loss