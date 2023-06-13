import os

import torch
import torch_geometric

from datasets.PowerFlowData import PowerFlowData
from networks.MPN import MPN
from utils.evaluation import load_model

from torch_geometric.loader import DataLoader
from utils.evaluation import evaluate_epoch
from utils.argument_parser import argument_parser

from utils.custom_loss_functions import Masked_L2_loss

LOG_DIR = 'logs'
SAVE_DIR = 'models'


@torch.no_grad()
def main():
    run_id = '20230609-4742'

    args = argument_parser()
    batch_size = args.batch_size
    grid_case = args.case
    loss_fn = Masked_L2_loss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    testset = PowerFlowData(root='data', case=grid_case,
                            split=[.5, .2, .3], task='test')
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    node_in_dim, node_out_dim, edge_dim = testset.get_data_dimensions()
    model = MPN(
        nfeature_dim=node_in_dim,
        efeature_dim=edge_dim,
        output_dim=node_out_dim,
        hidden_dim=16,
        n_gnn_layers=2,
        K=2,
        dropout_rate=0.5
    ).to(device)  # 40k params

    model, _ = load_model(model, run_id, device)

    # test_loss = evaluate_epoch(model, test_loader, loss_fn, device)
    # print(f"Test loss: {test_loss:.4f}")

    sample = testset[10].to(device)

    out = model(sample)
    
    out = out*sample.x[:,10:]
    input_x = sample.x[:,4:10]*sample.x[:,10:]
    # print(f"Input: {sample*testset.xystd + testset.xymean}")
    # print(f"Output: {out*testset.xystd + testset.xymean}")
    
    for i in range(sample.x.shape[0]):
        print("=====================================")
        print(f"Actual: {input_x[i,:]}")
        print(f"Predicted: {out[i,:]}")
        print(f"Difference: {input_x[i,:] - out[i,:]}")
    


if __name__ == "__main__":
    main()
