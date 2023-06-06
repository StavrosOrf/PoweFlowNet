import os

import torch
import torch_geometric

from datasets.PowerFlowData import PowerFlowData
from networks.MPN import MPN
from utils.evaluation import load_model

LOG_DIR = 'logs'
SAVE_DIR = 'models'

@torch.no_grad()
def main():
    run_id = '20230606-1194'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    testset = PowerFlowData(root='data', case='14', split=[.5, .2, .3], task='test')
    
    model = MPN(
        nfeature_dim=9, 
        efeature_dim=5, 
        output_dim=8, 
        hidden_dim=64, 
        n_gnn_layers=3, 
        K=3, 
        dropout_rate=0.5
    ).to(device) # 40k params
    
    model, _ = load_model(model, run_id, device)
    
    sample = testset[0].to(device)
    out = model(sample)
    print(f"Input: {sample}")
    print(f"Output: {out.shape}")
    
if __name__ == "__main__":
    main()