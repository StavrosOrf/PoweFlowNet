from typing import Callable, Optional, List, Tuple, Union
import os
import json

import torch
from torch_geometric.loader import DataLoader
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import torch.nn as nn
from tqdm import tqdm

from utils.custom_loss_functions import Masked_L2_loss, PowerImbalance


def append_to_json(log_path, run_id, result):
    log_entry = {str(run_id): result}

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    file_exists = os.path.isfile(log_path)
    try:
        with open(log_path, "r") as json_file:
            exist_log = json.load(json_file)
    except FileNotFoundError:
        exist_log = {}
    with open(log_path, "w") as json_file:
        exist_log.update(log_entry)
        json.dump(exist_log, json_file, indent=4)


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
    pbar = tqdm(loader, total=len(loader), desc='Training')
    for data in pbar:
        data = data.to(device) 
        optimizer.zero_grad()
        out = model(data)   # (N, 6), care about the first four. 
                            # data.y.shape == (N, 6)

        if isinstance(loss_fn, Masked_L2_loss):
            loss = loss_fn(out, data.y, data.x[:, 10:])
        elif isinstance(loss_fn, PowerImbalance):
            # have to mask out the non-predicted values, otherwise
            #   the network can learn to predict full-zeros
            masked_out = out*data.x[:, 10:] \
                        + data.x[:, 4:10]*(1-data.x[:, 10:])
            loss = loss_fn(masked_out, data.edge_index, data.edge_attr)
        else:
            loss = loss_fn(out, data.y)

        loss.backward()
        optimizer.step()
        num_samples += len(data)
        total_loss += loss.item() * len(data)

    mean_loss = total_loss / num_samples
    return mean_loss


def main():
    log_path = 'logs/save_logs.json'
    run_id = 'arb_id_01'
    result = {
        'train_loss': 0.3,
        'val_loss': 0.2,
    }
    append_to_json(log_path, run_id, result)


if __name__ == '__main__':
    main()
