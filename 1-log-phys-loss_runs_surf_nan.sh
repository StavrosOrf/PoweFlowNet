# for reference, this is one datapoint:
# Data(x=[14, 16], edge_index=[2, 20], edge_attr=[20, 5], y=[14, 6])

# this one gets ??? accuracy after 200 epochs
# WITH THIS ONE USE MODEL MPN
python3 test.py --cfg_json ./configs/standard.json\
                --round_experiments 1-log-phys-loss\
                --num-epochs 500\
                --data-dir ~/data/volume_2/power_flow_dataset\
                --batch-size 1024\
                --train_loss_fn mixed_mse_power_imbalance\
                --lr 0.1\
                --case 14v2\
                --model MaskEmbdMultiMPN\
                --save