"""This module provides functions for 
    - evaluation_epoch - evaluate performance over a whole epoch
    - other evaluation metrics function [NotImplemented]
"""
from typing import Callable, Optional, Union, Tuple
import os

import torch
from torch_geometric.loader import DataLoader
from torch.optim.optimizer import Optimizer
import torch.nn as nn
from tqdm import tqdm

from utils.custom_loss_functions import Masked_L2_loss, PowerImbalance, MixedMSEPoweImbalance, MaskedL2V2

LOG_DIR = 'logs'
SAVE_DIR = 'models'


def load_model(
    model: nn.Module,
    run_id: str,
    device: Union[str, torch.device]
) -> Tuple[nn.Module, dict]:
    SAVE_MODEL_PATH = os.path.join(SAVE_DIR, 'model_'+run_id+'.pt')
    if type(device) == str:
        device = torch.device(device)

    try:
        saved = torch.load(SAVE_MODEL_PATH, map_location=device)
        model.load_state_dict(saved['model_state_dict'])
    except FileNotFoundError:
        print("File not found. Could not load saved model.")
        return -1

    return model, saved


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
        device: str = 'cpu') -> float:
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
    pbar = tqdm(loader, total=len(loader), desc='Evaluating:')
    for data in pbar:
        data = data.to(device)
        out = model(data)

        if isinstance(loss_fn, Masked_L2_loss):
            loss = loss_fn(out, data.y, data.pred_mask)
        elif isinstance(loss_fn, PowerImbalance):
            # have to mask out the non-predicted values, otherwise
            #   the network can learn to predict full-zeros
            masked_out = out*data.pred_mask \
                        + data.pred_mask*(1-data.pred_mask)
            loss = loss_fn(masked_out, data.edge_index, data.edge_attr)
            # loss = loss_fn(data.y, data.edge_index, data.edge_attr)
        elif isinstance(loss_fn, MixedMSEPoweImbalance):
            loss = loss_fn(out, data.edge_index, data.edge_attr, data.y)
        else:
            loss = loss_fn(out, data.y)

        num_samples += len(data)
        total_loss += loss.item() * len(data)

    mean_loss = total_loss / num_samples
    return mean_loss

@torch.no_grad()
def evaluate_epoch_v2(
        model: nn.Module,
        loader: DataLoader,
        loss_fn: Callable,
        device: str = 'cpu') -> float:
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
    total_loss_terms = None
    num_samples = 0
    pbar = tqdm(loader, total=len(loader), desc='Evaluating:')
    for data in pbar:
        data = data.to(device)
        out = model(data)

        if isinstance(loss_fn, Masked_L2_loss):
            loss = loss_fn(out, data.y, data.pred_mask)
            loss_terms['total'] = loss
        elif isinstance(loss_fn, MaskedL2V2):
            loss_terms = loss_fn(out, data.y, data.pred_mask)
        elif isinstance(loss_fn, PowerImbalance):
            # have to mask out the non-predicted values, otherwise
            #   the network can learn to predict full-zeros
            masked_out = out*data.pred_mask \
                        + data.x*(1-data.pred_mask)
            loss = loss_fn(masked_out, data.edge_index, data.edge_attr)
            loss_terms['total'] = loss
            # loss = loss_fn(data.y, data.edge_index, data.edge_attr)
        elif isinstance(loss_fn, MixedMSEPoweImbalance):
            loss = loss_fn(out, data.edge_index, data.edge_attr, data.y)
            loss_terms['total'] = loss
        else:
            loss = loss_fn(out, data.y)
            loss_terms['total'] = loss

        num_samples += len(data)
        if total_loss_terms is None:
            total_loss_terms = {key: value.item() for key, value in loss_terms.items()}
        else:
            for key, value in total_loss_terms.items():
                total_loss_terms[key] = value + loss_terms[key].item() * len(data)

    mean_loss_terms = {key: value/num_samples for key, value in total_loss_terms.items()}
    return mean_loss_terms