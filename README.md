# PoweFlowNet
Leveraging Message Passing GNNs for High-Quality Power Flow Approximation.


![poweflownet_arch (1)](https://github.com/stavrosgreece/PoweFlowNet/assets/17108978/7ea2b68f-3aca-452c-a82a-b6614c70626f)

PoweFlowNet's distinctiveness, compared to existing PF GNN approaches, lies in its adept utilization of the capabilities from message-passing GNNs and high-order GCNs in a unique arrangement called PoweFlowConv, for handling a trainable masked embedding of the network graph. This innovative approach renders PoweFlowNet remarkably scalable, presenting an effective solution for the PF problem.

### Description

PoweFlowNet transforms the PF into a GNN node-regression problem by representing each bus as a node and each transmission line as an edge while maintaining the network's connectivity.

![PowerFlowProblemFOrmulation](https://github.com/stavrosgreece/PoweFlowNet/assets/17108978/dc4c2570-7148-497a-a02b-f6c550ad8ce7)

### Instructions


To train a model run train.py with the desired arguments. For example:
```
python3 train.py --cfg_json ./configs/standard.json\
                --num-epochs 2000\
                --data-dir ./data/
                --batch-size 128\
                --train_loss_fn mse_loss\
                --lr 0.001\
                --case 118v2\
                --model MaskEmbdMultiMPN\
                --save
```


### Datasets

Follow the links below to download the datasets and the trained models used in the paper.

Dataset link: https://tud365-my.sharepoint.com/:f:/g/personal/sorfanoudakis_tudelft_nl/EmWGZcpct51Gp2np1Zbv7NEBdANZCyFMlD7Iiyamp2_ztw?e=m4dNyK

Trained models link [temporary]: https://tud365-my.sharepoint.com/personal/nlin2_tudelft_nl/_layouts/15/onedrive.aspx?ct=1691140898356&or=Teams%2DHL&ga=1&LOF=1&id=%2Fpersonal%2Fnlin2%5Ftudelft%5Fnl%2FDocuments%2FProjects%2FPoweFlowNet&view=0



### File Structure
runnable files:
- `train.py` trains the model
- `results.py` plots the results

# Useful Information
First two dimensions out of seven in `edge_features` are `from_node` and `to_node`, and they are indexed from $1$. This is processed in the `PowerFlowData` dataset class. It is reindexed from $0$ and the `from_node` and `to_node` are removed from the `edge_features` tensor.

Raw data format: 
| Number | Description |
| --- | --- |
| N | number of nodes |
| E | number of edges |
| Fn = 9 | number of features per node |
| Fe = 5 | orginally 7, first two dims are `from_node` and `to_node` number of features per edge |
| Fn_out = 8 | number of output features per node |

| Tensor | Dimension |
| --- | --- |
| `Data.x` | (batch_size*N, Fe) |
| `Data.edge_index` | (2, E) |
| `Data.edge_attr` | (E, Fe) |
| `Data.y` | (batch_size*N, Fn) |


### Citation

If you use parts of this framework, datasets, or trained models, please cite as:
```
@misc{lin2023powerflownet,
      title={PowerFlowNet: Leveraging Message Passing GNNs for Improved Power Flow Approximation}, 
      author={Nan Lin and Stavros Orfanoudakis and Nathan Ordonez Cardenas and Juan S. Giraldo and Pedro P. Vergara},
      year={2023},
      eprint={2311.03415},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

