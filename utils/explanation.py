"""This module provides functions for 
    - explain_epoch - generate loss measurements for each node given k-hop subgraphs centered on that node 
    - some helper functions for the explain_epoch function
"""
from typing import Callable

import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils.convert import to_networkx
from torch_geometric.data import Data
import torch.nn as nn
from tqdm import tqdm
from utils.custom_loss_functions import Masked_L2_loss
from torch_geometric.utils.subgraph import k_hop_subgraph

import networkx as nx

LOG_DIR = 'logs'
SAVE_DIR = 'models'

@torch.no_grad()
def explain_epoch(
        model: nn.Module,
        loader: DataLoader,
        loss_fn: Callable,
        device: str = 'cpu',
        samples=16) -> float:
    """
    Generates loss measurements for each node given k-hop subgraphs centered on that node.

    Args:
        model (nn.Module): The trained neural network model to be evaluated.
        loader (DataLoader): The PyTorch Geometric DataLoader containing the evaluation data.
        loss_fn (nn.Module): The loss function to use to calculate losses
        device (str): The device used for evaluating the model (default: 'cpu').
        samples (int): The number of graphs to sample from the dataloader

    Returns:
        hop_losses: loss averaged over all nodes and batches, for each k-hop distance
        hop_losses_std: standard deviation of loss averaged over all nodes and batches, for each k-hop distance
        subgraph_nodes: for each node and k-hop distance, the size of the corresponding subgraph
        loss_gdata: for each node and k-hop distance, the average loss across batches
        nx_G: the networkx graph of the first graph returned by the dataloader, for plotting

    """
    model.eval()

    num_nodes, max_hopcount, nx_G = get_graphinfo(next(iter(loader)))

    losses = torch.zeros((num_nodes, max_hopcount+1))
    num_samples = torch.zeros((num_nodes, max_hopcount+1))
    subgraph_nnodes = torch.zeros((num_nodes, max_hopcount+1))

    pbar = tqdm(loader, total=len(loader))
    for iteration, data in enumerate(pbar):
        if iteration > samples:
            break

        data = data.to(device)

        ## to make the graph undirected
        # data.edge_index = torch.cat((data.edge_index, data.edge_index[[1, 0]]), dim=1)
        # data.edge_attr = torch.cat((data.edge_attr, data.edge_attr), dim=0)

        for n in range(num_nodes):

            for m in range(1, max_hopcount + 1):

                node_subset, _, _, edge_mask = k_hop_subgraph(torch.tensor([n]), m, data.edge_index,\
                                                                             relabel_nodes=False, directed=False)

                filtered_edgeattrs = data.edge_attr[edge_mask,:]
                filtered_edgeind = data.edge_index[:,edge_mask]

                masked_data = Data(x=data.x, y=data.y, edge_index=filtered_edgeind, edge_attr=filtered_edgeattrs, batch=data.batch)

                out = model(masked_data)

                if isinstance(loss_fn, Masked_L2_loss):
                    loss = loss_fn(out[n], data.y[n], data.x[:, 10:][n])
                else:
                    loss = loss_fn(out[n], data.y[n])

                losses[n,m] += loss.item()
                num_samples[n,m] += 1
                if iteration == 0:
                    # print(n, m, node_subset)
                    subgraph_nnodes[n,m] += node_subset.shape[0]

    return (losses / num_samples).mean(0), (losses / num_samples).std(0), subgraph_nnodes, (losses / num_samples), nx_G

def get_graphinfo(data: Data):
    G = to_networkx(data)
    # A = nx.adjacency_matrix(G).toarray()
    max_hopcount = max([max(j.values()) for (i,j) in nx.shortest_path_length(G)])
    num_nodes = data.x.shape[0]
    # masks = []
    # # for each node
    # # pbar = tqdm(range(num_nodes), total=num_nodes, position=1, leave=False)
    # for n in range(num_nodes):
    #     mask = []
    #     # for each hopcount n
    #     for c in range(1, max_hopcount+1):
    #         # run one hot with adjacency matrix n times
    #         x = np.zeros(num_nodes)
    #         x[n] = 1
    #         for _ in range(c):
    #             x = np.matmul(A, x) + x
    #         # the mask for each is the non-zero values in the previously one-hot vector
    #         mask.append(x != np.zeros(num_nodes)) # true means it's an n-hop neighbor
        
    #     node_mask = torch.nn.functional.one_hot(torch.tensor([n]), num_classes=num_nodes)
    #     neighborhood_mask = torch.from_numpy(np.array(mask))
    #     masks.append((node_mask, neighborhood_mask))

    return num_nodes, max_hopcount, G