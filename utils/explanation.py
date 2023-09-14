"""This module provides functions for 
    - explain_epoch - generate loss measurements for each node given k-hop subgraphs centered on that node 
    - some helper functions for the explain_epoch function
"""
from typing import Callable, Annotated, Sequence

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import numpy as np
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

@torch.no_grad() # TODO 
def explain_epoch(
        model: nn.Module,
        loader: DataLoader,
        loss_fn: Callable,
        device: str = 'cpu',
        num_batches=16) -> float:
    """
    Generates loss measurements for each node given k-hop subgraphs centered on that node. 
    
    Time complexity: O(num_nodes * diameter * num_batches)

    Args:
        model (nn.Module): The trained neural network model to be evaluated.
        loader (DataLoader): The PyTorch Geometric DataLoader containing the evaluation data.
        loss_fn (nn.Module): The loss function to use to calculate losses
        device (str): The device used for evaluating the model (default: 'cpu').
        num_batches (int): The number of graphs to sample from the dataloader

    Returns:
        loss_gdata: for each node and k-hop distance, the average loss across batches
        subgraph_nodes: for each node and k-hop distance, the size of the corresponding subgraph
        nx_G: the networkx graph of the first graph returned by the dataloader, for plotting

    """
    model.eval()

    num_nodes, diameter, nx_G = get_graphinfo(loader.dataset[0]) # assume all samples have the same graph, look at the first sample

    losses = torch.zeros((num_nodes, diameter+1))
    # later will create one subgraph for each node. 
    num_samples = torch.zeros((num_nodes, diameter+1)) # how many samples per subgraph are considered
    subgraph_nnodes = torch.zeros((num_nodes, diameter+1)) # how many nodes are in each subgraph 

    pbar = tqdm(loader, total=num_batches) # accepts batch_size >= 1
    for batch_idx, data in enumerate(pbar):
        if batch_idx > num_batches:
            break

        data = data.to(device)

        for node_idx in range(num_nodes):
            for m in range(0, diameter + 1):
                # Step 0: make subgraph
                bi_edge_index, bi_edge_attr = _make_bidirectional(data.edge_index, data.edge_attr)
                node_subset, edge_index, mapping, edge_mask = k_hop_subgraph(node_idx=node_idx, num_hops=m, 
                                                                             edge_index=bi_edge_index,
                                                                             num_nodes=num_nodes*len(data), # num_nodes * batch_size
                                                                             relabel_nodes=False, directed=False)

                filtered_edgeattrs = bi_edge_attr[edge_mask,:]
                filtered_edgeind = bi_edge_index[:,edge_mask]

                masked_data = Data(x=data.x, y=data.y, edge_index=filtered_edgeind, edge_attr=filtered_edgeattrs, batch=data.batch).to(device)

                # Step 1: inference model
                out = model(masked_data)

                if isinstance(loss_fn, Masked_L2_loss):
                    loss = loss_fn(out[node_idx], data.y[node_idx], data.x[:, 10:][node_idx]) # already averaged over samples
                else:
                    loss = loss_fn(out[node_idx], data.y[node_idx])

                losses[node_idx,m] += loss.item() # accumulated across batches
                num_samples[node_idx,m] += len(data) # += batch_size, count the total number of samples
                if batch_idx == 0:
                    # print(n, m, node_subset)
                    subgraph_nnodes[node_idx,m] += node_subset.shape[0]
    
    # return: (1) loss averaged over samples (2) number of nodes in each subgraph 
    # (3) networkx graph of the first sample
    return (losses / num_samples), subgraph_nnodes, nx_G

def get_graphinfo(data: Data) -> (int, int, nx.Graph):
    G = nx.from_edgelist(data.edge_index.T.tolist()) # NOTICE: actually has different results than torch_geometric.to_networkx
    diameter = nx.diameter(G)
    # diameter = max([max(j.values()) for (i,j) in nx.shortest_path_length(G)]) # equivalent to nx.diameter(G)
    num_nodes = len(G.nodes)

    return num_nodes, diameter, G

def _make_bidirectional(edge_index: torch.Tensor, edge_attr) -> (torch.Tensor, torch.Tensor):
    """
    Converts a directed edge index to an undirected edge index.

    Args:
        edge_index (torch.Tensor): A PyTorch Tensor of shape (2, num_edges) containing the edge index of a directed graph.

    Returns:
        torch.Tensor: A PyTorch Tensor of shape (2, 2*num_edges) containing the edge index of an undirected graph.

    """
    return torch.cat([edge_index, edge_index.flip([0])], dim=1), torch.cat([edge_attr, edge_attr.clone()], dim=0)

@plt.style.context('utils.small_fig')
def plot_num_nodes_subgraph(
    num_nodes_subgraph: Annotated[torch.tensor, 'num_nodes x diameter+1'],
    save_path: str = 'results/explain/num_nodes_subgraph.png',
    kwargs: dict = {}
) -> None:
    sorted, indices = torch.sort(num_nodes_subgraph[:,1:].sum(dim=1))

    fig, ax = plt.subplots(figsize=(10.5-30./72, 8))
    # plt. commands automatically call the current figure and axis
    plt.imshow(num_nodes_subgraph[indices], interpolation='nearest', aspect='auto', **kwargs)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Number of Nodes in Subgraph', size=30)
    xtick_loc = list(range(0, num_nodes_subgraph.shape[1], 2))
    if num_nodes_subgraph.shape[1] - 1 not in xtick_loc:
        xtick_loc.append(num_nodes_subgraph.shape[1] - 1)
    xtick_label = xtick_loc
    ax.set_xticks(xtick_loc, 
                  labels=xtick_label,
                  minor=False)
    ax.set_xticks(range(0, num_nodes_subgraph.shape[1]), minor=True)
    ytick_loc = np.linspace(0, num_nodes_subgraph.shape[0]-1, 7, endpoint=True, dtype=int)
    ytick_label = (ytick_loc+1).tolist()
    ax.set_yticks(ytick_loc, ytick_label, minor=False)
    plt.xlabel(r'Subgraph Size ($k$-hop)')
    plt.ylabel('Node Index')

    # print worst ones
    worst = [(int(x1), int(x2)) for x1, x2 in list(zip(num_nodes_subgraph[:,1:].sum(dim=1)[indices], indices))]
    print('subgsize,nidx')
    [print(x[0], "\t", x[1]) for x in worst[-16:]]
    plt.savefig(save_path, dpi=300)
    
    pass

@plt.style.context('utils.small_fig')
def plot_loss_subgraph(
    loss_subgraph: Annotated[torch.Tensor, 'num_nodes x diameter+1'],
    save_path: str = 'results/explain/loss_subgraph.png',
    kwargs: dict = {}
) -> None:
    mean = loss_subgraph.log().mean(dim=0).exp() # across nodes
    quantiles = {
        '1-sigma': {},
        '2-sigma': {},
        '3-sigma': {}
    }
    # 1-sigma quantiles, in between covers 68% of the data
    lower_quantile = torch.quantile(loss_subgraph, 0.16, dim=0)
    upper_quantile = torch.quantile(loss_subgraph, 0.84, dim=0)
    quantiles['1-sigma']['lower'] = lower_quantile
    quantiles['1-sigma']['upper'] = upper_quantile
    
    # 2-sigma quantiles, in between covers 95% of the data
    lower_quantile = torch.quantile(loss_subgraph, 0.025, dim=0)
    upper_quantile = torch.quantile(loss_subgraph, 0.975, dim=0)
    quantiles['2-sigma']['lower'] = lower_quantile
    quantiles['2-sigma']['upper'] = upper_quantile
    
    # 3-sigma quantiles, in between covers 99.7% of the data
    lower_quantile = torch.quantile(loss_subgraph, 0.00135, dim=0)
    upper_quantile = torch.quantile(loss_subgraph, 0.99865, dim=0)
    quantiles['3-sigma']['lower'] = lower_quantile
    quantiles['3-sigma']['upper'] = upper_quantile
    
    # transparencies for quantiles
    quantiles['1-sigma']['alpha'] = 0.4
    quantiles['2-sigma']['alpha'] = 0.2
    quantiles['3-sigma']['alpha'] = 0.1
    
    fig, ax = plt.subplots(figsize=(10.5, 8))
    plt.plot(range(mean.shape[0]), mean, color='C9', linewidth=2.5, **kwargs)
    for key, value in quantiles.items():
        lower_quantile = value['lower']
        upper_quantile = value['upper']
        alpha = value['alpha']
        plt.fill_between(range(mean.shape[0]), 
                         lower_quantile, upper_quantile, 
                         alpha=alpha,
                         color='C9',
                         **kwargs)
    ax.set_xlim([0, mean.shape[0]-1])
    ax.set_yscale('log')
    plt.title("Loss Per Subgraph")
    plt.xlabel(r'Subgraph Size ($k$-hop)')
    plt.ylabel('Loss')
    print(mean)
    
    plt.savefig(save_path, dpi=300)
    pass

@plt.style.context('utils.small_fig')
def plot_loss_subgraph_per_node(
    loss_subgraph: Annotated[torch.Tensor, 'num_nodes x diameter+1'],
    save_path: str = 'results/explain/loss_subgraph_per_node.png',
    kwargs: dict = {}
) -> None:
    sorted, indices = torch.sort(loss_subgraph[:,1:].sum(dim=1))

    fig, ax = plt.subplots(figsize=(10.5, 8))
    plt.imshow(loss_subgraph[indices], interpolation='nearest', aspect='auto',
               cmap='Blues', norm=mpl.colors.LogNorm())
    plt.xlabel(r'Subgraph Size ($k$-hop)')
    plt.ylabel('Node Index')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Loss')
    xtick_loc = list(range(0, loss_subgraph.shape[1], 2))
    if loss_subgraph.shape[1] - 1 not in xtick_loc:
        xtick_loc.append(loss_subgraph.shape[1] - 1)
    xtick_label = xtick_loc
    ax.set_xticks(xtick_loc, 
                  labels=xtick_label,
                  minor=False)
    ax.set_xticks(range(0, loss_subgraph.shape[1]), minor=True)
    ytick_loc = np.linspace(0, loss_subgraph.shape[0]-1, 7, endpoint=True, dtype=int)
    ytick_label = (ytick_loc+1).tolist()
    ax.set_yticks(ytick_loc, ytick_label, minor=False)
    
    # print worst ones
    worst = [(int(x1), int(x2)) for x1, x2 in list(zip(loss_subgraph[:,1:].sum(dim=1)[indices], indices))]
    print('loss,    nidx')
    [print(x[0], "\t", x[1]) for x in worst[-16:]]
    
    plt.savefig(save_path, dpi=300)
    pass

@plt.style.context('utils.small_fig')
def subplot_num_nodes_subgraph(
    num_nodes_subgraph_dict: dict[str, Annotated[torch.tensor, 'num_nodes x diameter+1']],
    save_path: str = 'results/explain/subplot_num_nodes_subgraph.png',
    kwargs: dict = {}
) -> None:
    fig = plt.figure(figsize=(0.75+9.25*len(num_nodes_subgraph_dict), 8))
    num_subplots = len(num_nodes_subgraph_dict)
    im_cbar_width_ratio = 6
    gspec = gridspec.GridSpec(nrows=1, ncols=im_cbar_width_ratio*num_subplots+1, figure=fig)
    
    # first len(num_nodes_subgraph_dict) axes, plot imshow
    im_axes = []
    for i, (key, num_nodes_subgraph) in enumerate(num_nodes_subgraph_dict.items()):
        ax = fig.add_subplot(gspec[0, im_cbar_width_ratio*i:im_cbar_width_ratio*(i+1)])
        im_axes.append(ax)
        sorted, indices = torch.sort(num_nodes_subgraph[:,1:].sum(dim=1))
        
        im_axes[i].imshow(num_nodes_subgraph[indices], interpolation='nearest', aspect='auto', **kwargs)
        xtick_loc = list(range(0, num_nodes_subgraph.shape[1], 2))
        if num_nodes_subgraph.shape[1] - 1 not in xtick_loc:
            xtick_loc.append(num_nodes_subgraph.shape[1] - 1)
        xtick_label = xtick_loc
        im_axes[i].set_xticks(xtick_loc, 
                      labels=xtick_label,
                      minor=False)
        im_axes[i].set_xticks(range(0, num_nodes_subgraph.shape[1]), minor=True)
        ytick_loc = np.linspace(0, num_nodes_subgraph.shape[0]-1, 7, endpoint=True, dtype=int)
        ytick_label = (ytick_loc+1).tolist()
        im_axes[i].set_yticks(ytick_loc, ytick_label, minor=False)
        im_axes[i].set_xlabel(r'Subgraph Size ($k$-hop)')
        
        if i == 0:
            im_axes[i].set_ylabel('Node Index')
    
    # last axes, plot colorbar
    cbar_ax = fig.add_subplot(gspec[0, -1])
    cbar = plt.colorbar(im_axes[i].images[0], 
                        cax=cbar_ax)
    cbar_tick_loc = np.linspace(0, 1, 6, endpoint=True).tolist()
    cbar_tick_label = [f'{x:.0%}' for x in cbar_tick_loc]
    cbar.ax.set_yticks(cbar_tick_loc[1:],
                        labels=cbar_tick_label[1:],
                        minor=False)
    cbar.ax.set_ylabel(f'Coverage of Subgraph', size=30)
            
    plt.savefig(save_path, dpi=300)
    pass
