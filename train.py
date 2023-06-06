from datetime import datetime

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from tqdm import tqdm

from datasets.PowerFlowData import PowerFlowData
from networks.MPN import MPN
from utils.argument_parser import argument_parser
from utils.training import train_epoch
from utils.evaluation import evaluate_epoch

def main():
    # Step 0: Parse Arguments and Setup
    args = argument_parser()
    num_epochs = args.num_epochs
    loss_fn = torch.nn.MSELoss()
    lr = 1e-3
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(1234)
    np.random.seed(1234)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    
    # Step 1: Load data
    trainset = PowerFlowData(root='data', case='14', split=[.5, .2, .3], task='train')
    valset = PowerFlowData(root='data', case='14', split=[.5, .2, .3], task='val')
    testset = PowerFlowData(root='data', case='14', split=[.5, .2, .3], task='test')
    train_loader = DataLoader(trainset, batch_size=32, shuffle=True)
    val_loader = DataLoader(valset, batch_size=32, shuffle=False)
    test_loader = DataLoader(testset, batch_size=32, shuffle=False)
    
    # Step 2: Create model and optimizer (and scheduler)
    model = MPN(
        nfeature_dim=9, 
        efeature_dim=5, 
        output_dim=8, 
        hidden_dim=64, 
        n_gnn_layers=3, 
        K=3, 
        dropout_rate=0.5
    ).to(device) # 40k params
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Step 3: Train model
    # TODO: Add checkpoint
    train_log = {
        'train': {
            'loss': []},
        'val': {
            'loss': []},
        'test': {
            'loss': []}
    }
    for epoch in tqdm(range(num_epochs)):
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss = evaluate_epoch(model, val_loader, loss_fn, device)
        test_loss = evaluate_epoch(model, test_loader, loss_fn, device)
        scheduler.step(val_loss)
        train_log['train']['loss'].append(train_loss)
        train_log['val']['loss'].append(val_loss)
        train_log['test']['loss'].append(test_loss)
        print(f"Epoch {epoch+1}/{num_epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, test_loss={test_loss:.4f}")
        
    # Step 4: Evaluate model
    test_loss = evaluate_epoch(model, test_loader, loss_fn, device)
    print(f"Final Test loss: {test_loss:.4f}")
    
    # Step 5: Save results
    torch.save(train_log, 'logs/train_log_'+datetime.now().strftime("%Y%m%d-%H%M%S")+'.pt')
    
    
if __name__ == '__main__':
    main()