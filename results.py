"""This program processes (and saves) results of the training. """
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

LOG_DIR = 'logs'

def main():
    run_id = '20230628-6515'
    TRAIN_LOG_PATH = os.path.join(LOG_DIR, 'train_log/train_log_'+run_id+'.pt')
    try:
        train_log = torch.load(TRAIN_LOG_PATH, map_location=torch.device('cpu'))
    except FileNotFoundError:
        print("File not found. terminating program.")
        return 1
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(train_log['train']['loss'], label='train')
    ax.plot(train_log['val']['loss'], label='val')
    if 'test' in train_log.keys():
        ax.plot(train_log['test']['loss'], label='test')
    ax.set_yscale('log')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (log scale)')
    ax.set_title('Loss vs Epoch')
    ax.legend()
    plt.savefig('results/'+run_id+'_loss_vs_epoch.png')
    plt.show()
    plt.close()
    
if __name__ == "__main__":
    main()