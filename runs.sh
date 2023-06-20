# for reference, this is one datapoint:
# Data(x=[14, 16], edge_index=[2, 20], edge_attr=[20, 5], y=[14, 6])

# this one gets 0.2595 accuracy after 20 epochs
# WITH THIS ONE USE MODEL MPN_simplenet
python train.py --num-epochs 50\
                --batch-size 128\
                --lr 0.001\
                --case 14\
                --nfeature_dim 16\
                --efeature_dim 5\
                --hidden_dim 64\
                --n_gnn_layers 2\
                --K 5\
                --dropout_rate 0.2\
                --model HigherOrderNN\
                --regularize=False\
                --regularization_coeff=0.2
