# for reference, this is one datapoint:
# Data(x=[14, 16], edge_index=[2, 20], edge_attr=[20, 5], y=[14, 6])

python3 train.py --cfg_json ./configs/standard.json\
                --num-epochs 2000\
                --data-dir data/power_flow_dataset/\
                --batch-size 128\
                --train_loss_fn mse_loss\
                --lr 0.001\
                --case 118\
                --model MaskEmbdMultiMPN\
                --save