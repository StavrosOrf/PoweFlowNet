"""This program processes (and saves) results of the training. """
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

log_dir = 'logs'

def main():
    try:
        files = os.listdir(log_dir)
        files = [file for file in files if file.endswith('.pt') and file.startswith('train_log')]
        latest_log_file = sorted(files)[-1]
        train_log = torch.load(os.path.join(log_dir, latest_log_file), map_location=torch.device('cpu'))
    except FileNotFoundError:
        print("File not found. terminating program.")
        return 1
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(train_log['train']['loss'], label='train')
    ax.plot(train_log['val']['loss'], label='val')
    ax.plot(train_log['test']['loss'], label='test')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss vs Epoch')
    ax.legend()
    plt.savefig('results/'+latest_log_file.split('.')[0]+'_loss_vs_epoch.png')
    plt.show()
    plt.close()
    
if __name__ == "__main__":
    main()