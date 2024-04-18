import os
import logging

import torch
import torch_geometric

from datasets.PowerFlowData import PowerFlowData
from networks.MPN import MPN, MPN_simplenet, SkipMPN, MaskEmbdMPN, MultiConvNet, MultiMPN, MaskEmbdMultiMPN
from utils.evaluation import load_model

from torch_geometric.loader import DataLoader
from utils.evaluation import evaluate_epoch, evaluate_epoch_v2
from utils.argument_parser import argument_parser

from utils.custom_loss_functions import Masked_L2_loss, PowerImbalance, MixedMSEPoweImbalance, MaskedL2V2

logger = logging.getLogger(__name__)

LOG_DIR = 'logs'
SAVE_DIR = 'models'


@torch.no_grad()
def main():
    run_id = '20240417-4378'
    # logging.basicConfig(filename=f'test_{run_id}.log', level=100)
    models = {
        'MPN': MPN,
        'MPN_simplenet': MPN_simplenet,
        'SkipMPN': SkipMPN,
        'MaskEmbdMPN': MaskEmbdMPN,
        'MultiConvNet': MultiConvNet,
        'MultiMPN': MultiMPN,
        'MaskEmbdMultiMPN': MaskEmbdMultiMPN
    }

    args = argument_parser()
    batch_size = args.batch_size
    grid_case = args.case
    data_dir = args.data_dir

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_param_path = os.path.join(data_dir, 'params', f'data_params_{run_id}.pt')
    
    data_param = torch.load(data_param_path, map_location='cpu')
    xymean, xystd = data_param['xymean'], data_param['xystd']
    edgemean, edgestd = data_param['edgemean'], data_param['edgestd']
        
    testset = PowerFlowData(root=data_dir, case=grid_case,
                            split=[.5, .2, .3], task='test',
                            xymean=xymean, xystd=xystd, edgemean=edgemean, edgestd=edgestd)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    pwr_imb_loss = PowerImbalance(*testset.get_data_means_stds()).to(device)
    mse_loss = torch.nn.MSELoss(reduction='mean').to(device)
    masked_l2 = Masked_L2_loss(regularize=False).to(device)
    all_losses = {
        'PowerImbalance': pwr_imb_loss,
        'Masked_L2_loss': masked_l2,
        'MSE': mse_loss,
    }
    
    
    # Network Parameters
    nfeature_dim = args.nfeature_dim
    efeature_dim = args.efeature_dim
    hidden_dim = args.hidden_dim
    output_dim = args.output_dim
    n_gnn_layers = args.n_gnn_layers
    conv_K = args.K
    dropout_rate = args.dropout_rate
    model = models[args.model]

    node_in_dim, node_out_dim, edge_dim = testset.get_data_dimensions()
    model = model(
        nfeature_dim=nfeature_dim,
        efeature_dim=efeature_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        n_gnn_layers=n_gnn_layers,
        K=conv_K,
        dropout_rate=dropout_rate,
    ).to(device)  # 40k params
    model.eval()

    model, _ = load_model(model, run_id, device)
    
    print(f"Model: {args.model}")
    print(f"Case: {grid_case}")
    
    _loss = MaskedL2V2()
    masked_l2_terms = evaluate_epoch_v2(model, test_loader, _loss, device)
    for key, value in masked_l2_terms.items():
        print(f"MaskedL2 {key}:\t{value:.4f}")
    for name, loss_fn in all_losses.items():
        test_loss = evaluate_epoch(model, test_loader, loss_fn, device)
        print(f"{name}:\t{test_loss:.4f}")
    


if __name__ == "__main__":
    main()
