from datetime import datetime
import os
import random

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from tqdm import tqdm

from datasets.PowerFlowData import PowerFlowData
from networks.GraphAutoEncoder import GraphAutoencoder
from utils.argument_parser import argument_parser
from utils.training import train_epoch, append_to_json
from utils.evaluation import evaluate_epoch
from utils.custom_loss_functions import Masked_L2_loss

import wandb


def main():
    # Step 0: Parse Arguments and Setup
    args = argument_parser()
    run_id = datetime.now().strftime("%Y%m%d") + '-' + str(random.randint(0, 9999))
    LOG_DIR = 'logs'
    SAVE_DIR = 'models'
    TRAIN_LOG_PATH = os.path.join(LOG_DIR, 'train_log/train_log_'+run_id+'.pt')
    SAVE_LOG_PATH = os.path.join(LOG_DIR, 'save_logs.json')
    SAVE_MODEL_PATH = os.path.join(SAVE_DIR, 'model_'+run_id+'.pt')

    # Training parameters
    data_dir = args.data_dir
    nfeature_dim = args.nfeature_dim
    num_epochs = args.num_epochs
    learning_rate = args.lr
    hidden_dim = args.hidden_dim
    loss_fn = Masked_L2_loss()
    latent_dim = 128
    batch_size = args.batch_size
    grid_case = args.case

    log_to_wandb = args.wandb
    if log_to_wandb:
        wandb.init(project="GraphAutoEncoder",
                   entity="GraphAutoEncoder",
                   name=run_id,
                   config=args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(1234)
    np.random.seed(1234)

    # Step 1: Load data
    trainset = PowerFlowData(root=data_dir, case=grid_case, split=[.5, .2, .3], task='train')
    valset = PowerFlowData(root=data_dir, case=grid_case, split=[.5, .2, .3], task='val')
    testset = PowerFlowData(root=data_dir, case=grid_case, split=[.5, .2, .3], task='test')
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    # Step 2: Create model and optimizer
    input_dim, _, _ = trainset.get_data_dimensions()
    model = GraphAutoencoder(input_dim, hidden_dim, latent_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Step 3: Train model
    best_train_loss = 10000.
    best_val_loss = 10000.
    saved_test_loss = 10000.
    train_log = {
        'train': {
            'loss': []},
        'val': {
            'loss': []},
        'test': {
            'loss': []}
    }

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss = evaluate_epoch(model, val_loader,loss_fn, device)
        test_loss = evaluate_epoch(model, test_loader,loss_fn, device)

        train_log['train']['loss'].append(train_loss)
        train_log['val']['loss'].append(val_loss)
        train_log['test']['loss'].append(test_loss)

        if log_to_wandb:
            wandb.log({'train_loss': train_loss,
                       'val_loss': val_loss, 'test_loss': test_loss})

        if train_loss < best_train_loss:
            best_train_loss = train_loss

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            saved_test_loss = test_loss

            if args.save:
                _to_save = {
                    'epoch': epoch,
                    'args': args,
                    'val_loss': best_val_loss,
                    'test_loss': test_loss,
                    'model_state_dict': model.state_dict(),
                }
                os.makedirs('models', exist_ok=True)
                torch.save(_to_save, SAVE_MODEL_PATH)

        print(f"Epoch {epoch+1} / {num_epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, test_loss={test_loss:.4f}, best_test_loss={saved_test_loss:.4f}, best_val_loss={best_val_loss:.4f}")

    print(f"Best validation loss: {best_val_loss:.4f}")

    # Step 4: Evaluate model
    if args.save:
        _to_load = torch.load(SAVE_MODEL_PATH)
        model.load_state_dict(_to_load['model_state_dict'])
        test_loss = evaluate_epoch(model, test_loader,loss_fn, device)
        print(f"Test loss: {best_val_loss:.4f}")

    # Step 5: Save results
    os.makedirs(os.path.join(LOG_DIR, 'train_log'), exist_ok=True)
    if args.save:
        append_to_json(
            SAVE_LOG_PATH,
            run_id,
            {
                'val_loss': f"{best_val_loss: .4f}",
                'test_loss': f"{saved_test_loss: .4f}",
                'train_log': TRAIN_LOG_PATH,
                'saved_file': SAVE_MODEL_PATH,
            }
        )
        torch.save(train_log, TRAIN_LOG_PATH)


if __name__ == '__main__':
    main()
