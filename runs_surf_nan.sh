# for reference, this is one datapoint:
# Data(x=[14, 16], edge_index=[2, 20], edge_attr=[20, 5], y=[14, 6])

# this one gets ??? accuracy after 200 epochs
# WITH THIS ONE USE MODEL MPN
python3 train.py --num-epochs 1000\
                --data-dir ~/data/volume_2/power_flow_dataset\
                --batch-size 128\
                --train_loss_fn power_imbalance\
                --lr 0.01\
                --case 118v2\
                --model MPN\
                --save
