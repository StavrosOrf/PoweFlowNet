import os
import torch
import numpy as np

def find_duplicates(tensor1, tensor2):
    # Step 1: Concatenate the two tensors
    combined = torch.cat([tensor1, tensor2], dim=0)
    
    # Step 2: Sort the combined tensor
    # Sort each row and then sort the rows lexicographically to bring duplicates together
    sorted_tensor, indices = combined.sort(dim=1)
    sorted_combined, sorted_indices = sorted_tensor.sort(dim=0)
    
    # Step 3: Identify duplicates by comparing each row with the next row
    duplicates = sorted_combined[1:] == sorted_combined[:-1]
    duplicate_rows = duplicates.all(dim=1)
    
    # Get indices of duplicates in the combined tensor
    duplicate_indices = sorted_indices[:-1][duplicate_rows]
    
    return duplicate_indices

def main():
    trn = torch.load('train_data_y.pt') # (N, 8)
    val = torch.load('val_data_y.pt')
    tst = torch.load('test_data_y.pt')
    
    dup_indices = find_duplicates(trn, tst)
    
    print(f'Number of total samples in train: {len(trn)}')
    print(f'Number of total samples in test: {len(tst)}')
    print(f'Number of duplicate samples: {len(dup_indices)}')
    pass

if __name__ == '__main__':
    main()