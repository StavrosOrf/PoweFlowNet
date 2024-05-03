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

feature_names_from_files = [
    'index',                # starting from 0 
    'type',                 # 
    'voltage magnitude',    # 
    'voltage angle degree', # 
    'Pd',                   # 
    'Qd',                   # 
    # 'Gs',                   # - equivalent to Pd, Qd                    8,
    # 'Bs',                   # - equivalent to Pd, Qd                    9,
    # 'Pg'                    # - removed
]

edge_feature_names_from_files = [
    'from_bus',             # 
    'to_bus',               #
    'r pu',                 # 
    'x pu',                 # 
]

def random_bus_type(data: Data) -> Data:
    " data.bus_type -> randomize "
    data.bus_type = torch.randint_like(data.bus_type, low=0, high=2)
    
    return data
    
def denormalize(input, mean, std):
    return input*(std.to(input.device)+1e-7) + mean.to(input.device)

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
        "edge_features.npy",
        "node_features.npy",
    ]
    split_order = {
        "train": 0,
        "val": 1,
        "test": 2
    }
    mixed_cases = [
        '118v2',
        '14v2',
    ]
    slack_mask = (0, 0, 1, 1) # 1 = need to predict, 0 = no need to predict
    gen_mask = (0, 1, 0, 1) 
    load_mask = (1, 1, 0, 0)
    bus_type_mask = (slack_mask, gen_mask, load_mask)

    def __init__(self, 
                root: str, 
                case: str = '14', 
                split: Optional[List[float]] = None, 
                task: str = "train", 
                transform: Optional[Callable] = None, 
                pre_transform: Optional[Callable] = None, 
                pre_filter: Optional[Callable] = None,
                normalize=True,
                xymean=None,
                xystd=None,
                edgemean=None,
                edgestd=None):

        assert len(split) == 3
        assert task in ["train", "val", "test"]
        self.normalize = normalize
        self.case = case  # THIS MUST BE EXECUTED BEFORE super().__init__() since it is used in raw_file_names and processed_file_names
        self.split = split
        self.task = task
        super().__init__(root, transform, pre_transform, pre_filter) # self.process runs here
        self.mask = torch.tensor([])
        # assign mean,std if specified
        if xymean is not None and xystd is not None:
            self.xymean, self.xystd = xymean, xystd
            print('xymean, xystd assigned.')
        else:
            self.xymean, self.xystd = None, None
        if edgemean is not None and edgestd is not None:
            self.edgemean, self.edgestd = edgemean, edgestd
            print('edgemean, edgestd assigned.')
        else:
            self.edgemean, self.edgestd = None, None
        self.data, self.slices = self._normalize_dataset(
            *torch.load(self.processed_paths[self.split_order[self.task]]))  # necessary, do not forget!

    def get_data_dimensions(self):
        return self[0].x.shape[1], self[0].y.shape[1], self[0].edge_attr.shape[1]

    def get_data_means_stds(self):
        assert self.normalize == True
        return self.xymean[:1, :], self.xystd[:1, :], self.edgemean[:1, :], self.edgestd[:1, :]

    def _normalize_dataset(self, data, slices, ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.normalize:
            # TODO an actual mask, perhaps never necessary though
            return data, slices

        # normalizing
        # for node attributes
        if self.xymean is None or self.xystd is None:
            xy = data.y # name 'xy' is from legacy. Shape (N, 4)
            mean = torch.mean(xy, dim=0, keepdim=True)
            std = torch.std(xy, dim=0, keepdim=True)
            self.xymean, self.xystd = mean, std
            # + 0.0000001 to avoid NaN's because of division by zero
        data.x = (data.x - self.xymean) / (self.xystd + 0.0000001)
        data.y = (data.y - self.xymean) / (self.xystd + 0.0000001)
        # for edge attributes
        if self.edgemean is None or self.edgestd is None:
            mean = torch.mean(data.edge_attr, dim=0, keepdim=True)
            std = torch.std(data.edge_attr, dim=0, keepdim=True)
            self.edgemean, self.edgestd = mean, std
        data.edge_attr = (data.edge_attr - self.edgemean) / (self.edgestd + 0.0000001)

        # deprecated: adding the mask
        # where x and y are unequal, the network must predict
        # 1 where value changed, 0 where it did not change
        # unequal = (data.x[:, 4:] != data.y).float()
        # data.prediction_mask = unequal
        # data.x = torch.concat([data.x, unequal], dim=1)

        return data, slices

    @property
    def raw_file_names(self) -> List[str]:
        if self.case != 'mixed':
            return ["case"+f"{self.case}"+"_"+name for name in self.partial_file_names]
        else:
            return ["case"+f"{case}"+"_"+name for case in self.mixed_cases for name in self.partial_file_names]

    @property
    def processed_file_names(self) -> List[str]:
        return [
            "case"+f"{self.case}"+"_processed_train.pt",
            "case"+f"{self.case}"+"_processed_val.pt",
            "case"+f"{self.case}"+"_processed_test.pt",
        ]

    def len(self):
        return self.slices['x'].shape[0]-1

    # def get(self, idx: int) -> Data: # override
    #     return self.data[idx]

    def process(self):
        # then use from_scipy_sparse_matrix()
        assert len(self.raw_paths) % 2 == 0
        raw_paths_per_case = [[self.raw_paths[i], self.raw_paths[i+1],] for i in range(0, len(self.raw_paths), 2)]
        all_case_data = [[],[],[]]
        for case, raw_paths in enumerate(raw_paths_per_case):
            # process multiple cases (if specified) e.g. cases = [14, 118]
            edge_features = torch.from_numpy(np.load(raw_paths[0])).float()
            node_features = torch.from_numpy(np.load(raw_paths[1])).float()

            assert self.split is not None
            if self.split is not None:
                split_len = [int(len(node_features) * i) for i in self.split]
            
            split_edge_features = torch.split(edge_features, split_len, dim=0)
            split_node_features = torch.split(node_features, split_len, dim=0)
            
            for idx in range(len(split_edge_features)):
                # shape of element in split_xx: [N, n_edges/n_nodes, n_features]
                # for each case, process train, val, test split
                y = split_node_features[idx][:, :, 2:] # shape (N, n_ndoes, 4); Vm, Va, P, Q
                bus_type = split_node_features[idx][:, :, 1].type(torch.long) # shape (N, n_nodes)
                bus_type_mask = torch.tensor(self.bus_type_mask)[bus_type] # shape (N, n_nodes, 4)
                x = y.clone()*(1.-bus_type_mask) # shape (N, n_nodes, 4)
                e = split_edge_features[idx] # shape (N, n_edges, 4)
                data_list = [
                    Data(
                        x=x[i],
                        y=y[i],
                        bus_type=bus_type[i],
                        pred_mask=bus_type_mask[i],
                        edge_index=e[i, :, 0:2].T.to(torch.long),
                        edge_attr=e[i, :, 2:],
                    ) for i in range(len(x))
                ]

                if self.pre_filter is not None:  # filter out some data
                    data_list = [data for data in data_list if self.pre_filter(data)]

                if self.pre_transform is not None:
                    data_list = [self.pre_transform(data) for data in data_list]
                    
                all_case_data[idx].extend(data_list)

        for idx, case_data in enumerate(all_case_data):
            data, slices = self.collate(case_data)
            torch.save((data, slices), self.processed_paths[idx])


def main():
    try:
        # shape = (N, n_edges, 7)       (from, to, ...)
        edge_features = np.load("data/raw/case14_edge_features.npy")
        # shape = (N, n_nodes, n_nodes)
        adj_matrix = np.load("data/raw/case14_adjacency_matrix.npy")
        # shape = (N, n_nodes, 9)
        node_features_x = np.load("data/raw/case14_node_features_x.npy")
        # shape = (N, n_nodes, 8)
        node_features_y = np.load("data/raw/case14_node_features_y.npy")
    except FileNotFoundError:
        print("File not found.")

    print(f"edge_features.shape = {edge_features.shape}")
    print(f"adj_matrix.shape = {adj_matrix.shape}")
    print(f"node_features_x.shape = {node_features_x.shape}")
    print(f"node_features_y.shape = {node_features_y.shape}")

    trainset = PowerFlowData(root="data", case=14,
                             split=[.5, .2, .3], task="train")
    train_loader = torch_geometric.loader.DataLoader(
        trainset, batch_size=12, shuffle=True)
    print(len(trainset))
    print(trainset[0])
    print(next(iter(train_loader)))
    pass


if __name__ == "__main__":
    main()