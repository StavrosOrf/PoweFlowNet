# PowerFlowNet
Leveraging Message Passing GNNs for High-Quality Power Flow Approximation.

![image](https://github.com/StavrosOrf/PoweFlowNet/assets/17108978/1a6398c5-cac6-40cf-a3a1-0bc8fb66a0dc)


PowerFlowNet's distinctiveness, compared to existing PF GNN approaches, lies in its adept utilization of the capabilities from message-passing GNNs and high-order GCNs in a unique arrangement called PowerFlowConv, for handling a trainable masked embedding of the network graph. This innovative approach renders PoweFlowNet remarkably scalable, presenting an effective solution for the PF problem.

### Description

PowerFlowNet transforms the PF into a GNN node-regression problem by representing each bus as a node and each transmission line as an edge while maintaining the network's connectivity.

![image](https://github.com/StavrosOrf/PoweFlowNet/assets/17108978/3c3314c8-c111-41a7-8eb6-2116533f7f72)


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

[Dataset link](https://surfdrive.surf.nl/files/index.php/s/Qw4RHLvI2RPBIBL)

[Trained models link](https://surfdrive.surf.nl/files/index.php/s/iunfVTGsABT5NaD)



### File Structure
runnable files:
- `train.py` trains the model
- `results.py` plots the results
- and more scripts to generate results and plots ...

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

