"""
this file defines the class of PowerFlowData, which is used to load the data of Power Flow
"""
import os
from typing import Callable, Optional, List, Tuple, Union

import torch
import numpy as np
import torch.utils.data as data
import torch_geometric
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import from_scipy_sparse_matrix, dense_to_sparse
import matplotlib.pyplot as plt

from torch_geometric.datasets import Planetoid

class PowerFlowData(InMemoryDataset):
    """PowerFlowData(InMemoryDataset)

    Parameters:
        root (str, optional) – Root directory where the dataset should be saved. (optional: None)
        pre_filter (callable)- A function 
        
    Comments:
        we actually do not need adjacency matrix, since we can use edge_index to represent the graph from `edge_features`

    Returns:
        class instance of PowerFlowData
    """
    partial_file_names = [
        "adjacency_matrix.npy",
        "edge_features.npy",
        "node_features_x.npy",
        "node_features_y.npy"
    ]
    split_order = {
        "train": 0,
        "val": 1,
        "test": 2
    }
    def __init__(self, 
                root: str, 
                case: str = '14', 
                split: Optional[List[float]] = None, 
                task: str = "train", 
                transform: Optional[Callable] = None, 
                pre_transform: Optional[Callable] = None, 
                pre_filter: Optional[Callable] = None):
        assert len(split) == 3
        assert task in ["train", "val", "test"]
        self.case = case # THIS MUST BE EXECUTED BEFORE super().__init__() since it is used in raw_file_names and processed_file_names
        self.split = split
        self.task = task
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0]) # necessary, do not forget!
        
    @property
    def raw_file_names(self) -> List[str]:
        return ["case"+f"{self.case}"+"_"+name for name in self.partial_file_names]
    
    @property
    def processed_file_names(self) -> List[str]:
        return ["case"+f"{self.case}"+"_processed_data.pt"]
    
    def len(self):
        return self.slices['x'].shape[0]-1
    
    # def get(self, idx: int) -> Data: # override
    #     return self.data[idx]
        
    
    def process(self):
        adj_mat = dense_to_sparse(torch.from_numpy(np.load(self.raw_paths[0]))) # then use from_scipy_sparse_matrix()
        edge_features = torch.from_numpy(np.load(self.raw_paths[1])).float()
        node_features_x = torch.from_numpy(np.load(self.raw_paths[2])).float()
        node_features_y = torch.from_numpy(np.load(self.raw_paths[3])).float()
        
        if self.split is not None:
            split_len = [int(len(node_features_x) * i) for i in self.split]
            edge_features = torch.split(edge_features, split_len, dim=0)[self.split_order[self.task]]
            node_features_x = torch.split(node_features_x, split_len, dim=0)[self.split_order[self.task]]
            node_features_y = torch.split(node_features_y, split_len, dim=0)[self.split_order[self.task]]
        
        data_list = [
            Data(
                x=node_features_x[i],
                y=node_features_y[i],
                edge_index=edge_features[i, :, 0:2].T.to(torch.long)-1,
                edge_attr=edge_features[i, :, 2:],
            ) for i in range(len(node_features_x))
        ]

        if self.pre_filter is not None: # filter out some data
            data_list = [data for data in data_list if self.pre_filter(data)]
        
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
            
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
    
def main():
    try:
        edge_features = np.load("data/raw/case14_edge_features.npy")        # shape = (N, n_edges, 7)       (from, to, ...)
        adj_matrix = np.load("data/raw/case14_adjacency_matrix.npy")        # shape = (N, n_nodes, n_nodes) 
        node_features_x = np.load("data/raw/case14_node_features_x.npy")    # shape = (N, n_nodes, 9)
        node_features_y = np.load("data/raw/case14_node_features_y.npy")    # shape = (N, n_nodes, 8)
    except FileNotFoundError:
        print("File not found.")
    
    print(f"edge_features.shape = {edge_features.shape}")
    print(f"adj_matrix.shape = {adj_matrix.shape}")
    print(f"node_features_x.shape = {node_features_x.shape}")
    print(f"node_features_y.shape = {node_features_y.shape}")
    
    trainset = PowerFlowData(root="data", case=14, split=[.5, .2, .3], task="train")
    train_loader = torch_geometric.loader.DataLoader(trainset, batch_size=12, shuffle=True)
    print(len(trainset))
    print(trainset[0])
    print(next(iter(train_loader)))
    pass

if __name__ == "__main__":
    main()