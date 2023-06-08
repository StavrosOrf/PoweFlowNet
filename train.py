from datetime import datetime
import os
import random

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from tqdm import tqdm

from datasets.PowerFlowData import PowerFlowData, select_features
from networks.MPN import MPN
from utils.argument_parser import argument_parser
from utils.training import train_epoch, append_to_json
from utils.evaluation import evaluate_epoch

def main():
    # Step 0: Parse Arguments and Setup
    args = argument_parser()
    run_id = datetime.now().strftime("%Y%m%d") + '-' + str(random.randint(0, 9999))
    LOG_DIR = 'logs'
    SAVE_DIR = 'models'
    TRAIN_LOG_PATH = os.path.join(LOG_DIR, 'train_log/train_log_'+run_id+'.pt')
    SAVE_LOG_PATH = os.path.join(LOG_DIR, 'save_logs.json')
    SAVE_MODEL_PATH = os.path.join(SAVE_DIR, 'model_'+run_id+'.pt')
    
    num_epochs = args.num_epochs
    
    loss_fn = torch.nn.MSELoss()
    lr = 1e-3
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(1234)
    np.random.seed(1234)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    
    # Step 1: Load data
    trainset = PowerFlowData(root='~/data/volume_2/power_flow_dataset', case='14', split=[.5, .2, .3], task='train', 
                             transform=select_features((2,3,4,5), (2,3,4,5)))
    valset = PowerFlowData(root='~/data/volume_2/power_flow_dataset', case='14', split=[.5, .2, .3], task='val', 
                           transform=select_features((2,3,4,5), (2,3,4,5)))
    testset = PowerFlowData(root='~/data/volume_2/power_flow_dataset', case='14', split=[.5, .2, .3], task='test', 
                            transform=select_features((2,3,4,5), (2,3,4,5)))
    train_loader = DataLoader(trainset, batch_size=128, shuffle=True)
    val_loader = DataLoader(valset, batch_size=128, shuffle=False)
    test_loader = DataLoader(testset, batch_size=128, shuffle=False)
    
    # Step 2: Create model and optimizer (and scheduler)
    model = MPN(
        nfeature_dim=4, 
        efeature_dim=5, 
        output_dim=4, 
        hidden_dim=64, 
        n_gnn_layers=6, 
        K=3, 
        dropout_rate=0.5
    ).to(device) # 40k params
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                           mode='min', 
                                                           factor=0.5, 
                                                           patience=5,
                                                           verbose=True)
    
    # Step 3: Train model
    # TODO: Add checkpoint
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
    for epoch in tqdm(range(num_epochs), position=0, leave=True):
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss = evaluate_epoch(model, val_loader, loss_fn, device)
        test_loss = evaluate_epoch(model, test_loader, loss_fn, device)
        scheduler.step(val_loss)
        train_log['train']['loss'].append(train_loss)
        train_log['val']['loss'].append(val_loss)
        train_log['test']['loss'].append(test_loss)
        print(f"Epoch {epoch+1}/{num_epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, test_loss={test_loss:.4f}")
        
        # if epoch >= 8 * num_epochs // 10:
        if True:
            # if val_loss < 0.95 * best_val_loss:
            if True:
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
        
    # Step 4: Evaluate model
    test_loss = evaluate_epoch(model, test_loader, loss_fn, device)
    print(f"Final Test loss: {test_loss:.4f}")
    
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