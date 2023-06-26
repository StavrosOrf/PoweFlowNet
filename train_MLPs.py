from datetime import datetime
import os
import random

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from tqdm import tqdm

from datasets.PowerFlowData import PowerFlowData
from networks.MLP import MLP
from utils.argument_parser import argument_parser
from utils.training import train_epoch, append_to_json
from utils.evaluation import evaluate_epoch
from utils.custom_loss_functions import Masked_L2_loss

""" 
This script is used to train simple MLPs on the power flow problem, so the results can be used as baseline.
"""
cases = ['case14', 'case118', 'case6470rte']
# cases = ['case6470rte']

args = argument_parser()
num_epochs = args.num_epochs
data_dir = args.data_dir
num_epochs = args.num_epochs
loss_fn = Masked_L2_loss()
lr = args.lr
batch_size = args.batch_size
print(f'Bath size: {batch_size}')


for case in cases:
    case_name = case.split("case")[1]
    print(f'\n\nCase {case_name} MLP is trained...')

    eval_loss_fn = Masked_L2_loss(regularize=False)

    # Step 1: Load data
    trainset = PowerFlowData(
        root=data_dir, case=case_name, split=[.5, .2, .3], task='train')
    valset = PowerFlowData(root=data_dir, case=case_name,
                           split=[.5, .2, .3], task='val')
    testset = PowerFlowData(root=data_dir, case=case_name,
                            split=[.5, .2, .3], task='test')
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    # Step 2: Define model
    num_inputs = trainset[0].x.shape[0] * trainset[0].x.shape[1]
    num_outputs = trainset[0].y.shape[0] * trainset[0].y.shape[1]

    print(f'Number of inputs: {num_inputs}| Number of outputs: {num_outputs}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLP(num_inputs, num_outputs, 128, 3, 0.2)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=num_epochs)

    # Step 3: Train model
    best_train_loss = 10000.
    best_val_loss = 10000.

    train_losses = []
    val_losses = []
    SAVE_MODEL_PATH = f'./models/testing/mlp_{case_name}.pt'

    for epoch in tqdm(range(num_epochs)):
        train_loss = train_epoch(
            model, train_loader, eval_loss_fn, optimizer, device)
        val_loss = evaluate_epoch(model, val_loader, eval_loss_fn, device)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_train_loss = train_loss
            _to_save = {
                'epoch': epoch,
                'args': args,
                'val_loss': best_val_loss,
                'model_state_dict': model.state_dict(),
            }
            os.makedirs('models', exist_ok=True)
            torch.save(_to_save,SAVE_MODEL_PATH)

        print(f'Epoch {epoch+1}/{num_epochs}: Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f} best val loss: {best_val_loss:.4f} ')
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    # Step 4: Evaluate model
    model.load_state_dict(torch.load(f'./models/testing/mlp_{case_name}.pt'))
    test_loss = evaluate_epoch(model, test_loader, loss_fn, device)
