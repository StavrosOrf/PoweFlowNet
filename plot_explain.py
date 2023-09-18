r"Plot only the saved explanation results."

import argparse
from argparse import ArgumentParser
import os

import torch

from utils.explanation import subplot_loss_subgraph, subplot_loss_subgraph_per_node, subplot_num_nodes_subgraph

all_run_id = {
    '14v2': '20230627-576',
    '118v2': '20230627-9288',
    '6470rtev2': '20230627-1251',
}

def main():
    # Parameters
    result_dir = 'results/explain/'

    # Load processed explanation
    all_num_nodes_subgraph = {
        key: None for key in all_run_id.keys()
    }
    all_loss_subgraph = {
        key: None for key in all_run_id.keys()
    }
    
    for grid_case, run_id in all_run_id.items():
        try:
            loss_subgraph = torch.load(os.path.join(result_dir, run_id, 'loss_subgraph'+'_case_'+grid_case+'.pt'))
            num_nodes_subgraph = torch.load(os.path.join(result_dir, run_id, 'num_nodes_subgraph'+'_case_'+grid_case+'.pt'))
            all_num_nodes_subgraph[grid_case] = num_nodes_subgraph/num_nodes_subgraph.max()
            all_loss_subgraph[grid_case] = loss_subgraph            
        except FileNotFoundError:
            print(f'File not found. run_id: {run_id}, grid_case: {grid_case}')
            return 1   
    
    # Plot
    subplot_num_nodes_subgraph(all_num_nodes_subgraph, save_path=os.path.join(result_dir, 'subplot_num_nodes_subgraph'+'.pdf'))
    subplot_loss_subgraph(all_loss_subgraph, save_path=os.path.join(result_dir, 'subplot_loss_subgraph'+'.pdf'))
    subplot_loss_subgraph_per_node(all_loss_subgraph, save_path=os.path.join(result_dir, 'subplot_loss_subgraph_per_node'+'.pdf'))

if __name__ == '__main__':
    main()