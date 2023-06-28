import torch
from datasets.PowerFlowData import PowerFlowData
from networks.MPN import MPN,MPN_simplenet
from utils.custom_loss_functions import Masked_L2_loss
import time
from utils.argument_parser import argument_parser
from pygsp import graphs
import numpy as np
from collaborative_filtering import tikhonov_regularizer,collaborative_filtering_testing

""" 
This script is used to evaluate the performance of various models on the power flow problem.
Models:
    - MPN
    - Tikhonov Regularizer
    - MLP
    - Newton-Raphson method
"""
cases = ['case14','case118','case6470rte']
# cases = ['case6470rte']

for case in cases:

    case_name = case.split("case")[1]
    print(f'\n\nCase {case_name} is being evaluated...')
    #Load testing data
    testset = PowerFlowData(root="./data/", case=case_name, split=[.5, .2, .3], task='test')
    sample_number = 1000
    if sample_number > len(testset):
        sample_number = len(testset)
    print(f'Number of samples: {sample_number}')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    eval_loss_fn = Masked_L2_loss(regularize=False)

    #Load MPN model
    model_path = "./models/testing/mpn_" + case_name + ".pt"

    MPN_model = MPN(
        nfeature_dim=6,
        efeature_dim=5,
        output_dim=6,
        hidden_dim=129,
        n_gnn_layers=4,
        K=3,
        dropout_rate=0.2
    ).to(device)

    _to_load = torch.load(model_path)
    MPN_model.load_state_dict(_to_load['model_state_dict'])
    MPN_model.eval()

    #Get loss of MPN model and execution time    
    timer_MPN = 0
    loss_MPN = 0
        
    for i, sample in enumerate(testset[:sample_number]):        
        time_start_gnn = time.time()
        result = MPN_model(sample.to(device))
        time_end_gnn = time.time()
        loss_MPN += eval_loss_fn(result, sample.y.to(device), sample.x[:, 10:].to(device)).item()

        timer_MPN += time_end_gnn - time_start_gnn

    
    print(f'Loss of MPN model: {loss_MPN/sample_number}')
    print(f'Execution time of MPN model: {timer_MPN/sample_number}')


    ###### Tikhonov Regularizer ##########################
    # Load adjacency matrix from file
    file_path = "./data/raw/case" + str(case_name) + '_adjacency_matrix.npy'
    adjacency_matrix = np.load(file_path)
    # print(adjacency_matrix.shape)

    num_of_nodes = adjacency_matrix.shape[0]
    # print(f'Number of nodes: {num_of_nodes}')

    # create graph from adjacency matrix
    G = graphs.Graph(adjacency_matrix)

    # get incidence matrix
    G.compute_differential_operator()
    B = G.D.toarray()
    # print(f'B: {B.shape}')
    # get laplacian matrix
    L = G.L.toarray()
    # print(f'Laplacian: {L.shape}')

    timer_regularizer = 0
    loss_MPN = 0
        
    for i, sample in enumerate(testset[:sample_number]):        
        time_start = time.time()                
        result = tikhonov_regularizer(1.25, L, sample.x[:,4:8], sample.x[:, 10:].to(device))  
        # result = collaborative_filtering_testing(sample.x[:,4:8], sample.x[:, 10:14], B,sample.y[:,:4],4)   
        time_end = time.time()
        loss_MPN += eval_loss_fn(result, sample.y[:,:4], sample.x[:, 10:14].to(device)).item()
    
        timer_regularizer += time_end - time_start

    print(f'Loss of Tikhonov Regularizer: {loss_MPN/sample_number}')
    print(f'Execution time of Tikhonov Regularizer: {timer_regularizer/sample_number}')
    
    ###### MLP ##########################
    



