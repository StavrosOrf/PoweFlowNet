import argparse
from argparse import ArgumentParser
import os

from networks.MPN import MPN_simplenet, MaskEmbdMultiMPN
from datasets.PowerFlowData import PowerFlowData
from torch_geometric.loader import DataLoader
from utils.evaluation import load_model
from utils.explanation import explain_epoch, plot_loss_subgraph, plot_loss_subgraph_per_node, plot_num_nodes_subgraph
from utils.custom_loss_functions import Masked_L2_loss
import torch
import matplotlib.pyplot as plt
import networkx as nx

parser = ArgumentParser()
parser.add_argument('--load', default=False, action=argparse.BooleanOptionalAction, help='load model')
parser.add_argument('--num_batches', default=10, type=int, help='number of batches to evaluate')
parser.add_argument('--batch_size', '-bs', default=128, type=int, help='batch size')
args = parser.parse_args()

def main():
    # Parameters
    run_id = '20230627-9288' # mse, or 20230627-5949, mixed
    data_dir = '/home/nlin/data/volume_2/power_flow_dataset'
    grid_case = '118v2'
    num_batches = args.num_batches
    batch_size = args.batch_size
    result_dir = 'results/explain/'+run_id
    if args.load:
        try:
            loss_subgraph = torch.load(os.path.join(result_dir, 'loss_subgraph'+'_case_'+grid_case+'.pt'))
            num_nodes_subgraph = torch.load(os.path.join(result_dir, 'num_nodes_subgraph'+'_case_'+grid_case+'.pt'))
        except FileNotFoundError:
            print('File not found. Please run without --load first.')
            return 1   
    else:
        # Load model, loss function and data
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = MaskEmbdMultiMPN(
            nfeature_dim=6,
            efeature_dim=5,
            output_dim=6,
            hidden_dim=129,
            n_gnn_layers=4,
            K=3,
            dropout_rate=0.2
        ).to(device)
        model.eval()
        model, _ = load_model(model, run_id, device)

        eval_loss_fn = Masked_L2_loss(regularize=False).to(device)

        testset = PowerFlowData(root=data_dir, case=grid_case,
                                    split=[.5, .2, .3], task='test')
        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
        
        # Calculate loss per subgraph
        loss_subgraph, num_nodes_subgraph, nx_G = explain_epoch(model, test_loader, eval_loss_fn, device=device, num_batches=num_batches)
        
        os.makedirs(result_dir, exist_ok=True)
        torch.save(loss_subgraph, os.path.join(result_dir, 'loss_subgraph'+'_case_'+grid_case+'.pt'))
        torch.save(num_nodes_subgraph, os.path.join(result_dir, 'num_nodes_subgraph'+'_case_'+grid_case+'.pt'))
    
    # Plot loss per subgraph
    plot_num_nodes_subgraph(num_nodes_subgraph, save_path=os.path.join(result_dir, 'num_nodes_subgraph'+'_case_'+grid_case+'.png'))
    plot_loss_subgraph(loss_subgraph, save_path=os.path.join(result_dir, 'loss_subgraph'+'_case_'+grid_case+'.png'))
    plot_loss_subgraph_per_node(loss_subgraph, save_path=os.path.join(result_dir, 'loss_subgraph_per_node'+'_case_'+grid_case+'.png'))
    

if __name__ == '__main__':
    main()