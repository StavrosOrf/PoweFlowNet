"""
This script is used to generate error per feature plots and statistics for the MPN model.
"""

import os

import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import matplotlib.colors as mcolors
import numpy as np
import time

from datasets.PowerFlowData import PowerFlowData
from networks.MPN import MaskEmbdMultiMPN
from utils.custom_loss_functions import Masked_L2_loss

LOG_DIR = 'logs'
SAVE_DIR = 'models'

os.makedirs('./results', exist_ok=True)

feature_names_output = [
    'Voltage Magnitude (p.u.)',    # --- we care about this
    'Voltage Angle (deg)',        # --- we care about this
    'Active Power (MW)',         # --- we care about this
    'Reactive Power (MVar)',       # --- we care about this
    'Gs',                   # -
    'Bs'                    # -
]

GET_RESULTS = False
sample_number = 200000
cases = ['case14', 'case118', 'case6470rte']
# cases = ['case6470rte']

if GET_RESULTS:

    for case in cases:

        case_name = case.split("case")[1]

        print(f'\n\nCase {case_name} is being evaluated...')
        # Load testing data
        testset = PowerFlowData(root="./data/", case=case_name,
                                split=[.5, .2, .3], task='test')

        if sample_number > len(testset):
            sample_number = len(testset)
        print(f'Number of samples: {sample_number}')

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # device = torch.device("cuda:0")

        eval_loss_fn = Masked_L2_loss(regularize=False)

        # Load MPN model
        model_path = "./models/testing/mpn_" + case_name + ".pt"

        MPN_model = MaskEmbdMultiMPN(
            nfeature_dim=6,
            efeature_dim=5,
            output_dim=6,
            hidden_dim=129,
            n_gnn_layers=4,
            K=3,
            dropout_rate=0.2
        ).to(device)

        _to_load = torch.load(model_path, map_location=device)
        MPN_model.load_state_dict(_to_load['model_state_dict'])
        MPN_model.eval()

        # Get loss of MPN model and execution time
        timer_MPN = 0
        loss_MPN = 0

        preds = []
        targets = []
        masks = []
        types = []

        for i, sample in enumerate(testset[:sample_number]):
            time_start_gnn = time.time()
            result = MPN_model(sample.to(device))

            preds.append(result.detach().cpu())
            targets.append(sample.y.detach().cpu())
            masks.append(sample.x[:, 10:].detach().cpu())
            types.append(
                np.argmax(np.array(sample.x[:, :4].detach().cpu()), axis=1))

            time_end_gnn = time.time()
            loss_MPN += eval_loss_fn(result, sample.y.to(device),
                                     sample.x[:, 10:].to(device)).item()

            timer_MPN += time_end_gnn - time_start_gnn

        print(f'Loss of MPN model: {loss_MPN/sample_number}')
        print(f'Execution time of MPN model: {timer_MPN/sample_number}')

        mean = testset.xymean[0].detach().cpu()
        std = testset.xystd[0].detach().cpu()

        preds = torch.stack(preds, dim=0) * std + mean
        targets = torch.stack(targets, dim=0) * std + mean
        error = preds - targets

        error = error.detach().cpu().numpy()
        masks = torch.stack(masks, dim=0).detach().cpu().numpy()
        types = np.array(types)

        masks = masks[:, :, :4]
        errors = error[:, :, :4]
        # Save results
        print(f'masks shape: {masks.shape}')
        print(f'types shape: {types.shape}')
        print(f'error shape: {errors.shape}')

        with open('./results/'+case_name+'_masks.npy', 'wb') as f:
            np.save(f, masks)
        with open('./results/'+case_name+'_types.npy', 'wb') as f:
            np.save(f, types)
        with open('./results/'+case_name+'_errors.npy', 'wb') as f:
            np.save(f, errors)


# Plot results
# cases = ['case14']
plt.rcParams['font.family'] = ['serif']
plt.subplots(3, 4,
             figsize=(10, 7))
#     tight_layout=True,)
# sharey=True,
# sharex=True)
plot_counter = 0

for counter_i, case in enumerate(cases):
    case_name = case.split("case")[1]
    # load results
    number = 1000000
    errors = np.load('./results/'+case_name+'_errors.npy')[:number, :, :]
    masks = np.load('./results/'+case_name+'_masks.npy')[:number, :]
    types = np.load('./results/'+case_name+'_types.npy')[:number]
    # print(types)

    # get number of 1 in masks
    print(f'Number of Voltage Magnitude: {np.sum(masks[0,:,0]==1)}')
    n_vm = np.sum(masks[0, :, 0] == 1)
    print(f'Number of Voltage Angle: {np.sum(masks[0,:,1]==1)}')
    n_va = np.sum(masks[0, :, 1] == 1)
    print(f'Number of Active Power: {np.sum(masks[0,:,2]==1)}')
    n_ap = np.sum(masks[0, :, 2] == 1)
    print(f'Number of Reactive Power: {np.sum(masks[0,:,3]==1)}')
    n_rp = np.sum(masks[0, :, 3] == 1)

    # multiply errors by masks
    # replace zeros in mask with 1
    masks[masks == 0] = 0.00001
    errors = errors*masks

    print(f'Number of Loads: {np.sum(types[0,:]==2)}')
    n_loads = np.sum(types[0, :] == 2)
    print(f'Number of Generators: {np.sum(types[0,:]==1)}')
    n_gens = np.sum(types[0, :] == 1)
    print("="*80)   

    #print the average and standard deviation of errors for each feature only when mask is 1
    indexes = np.where(masks[0, :, 0] == 1)[0]    
    vm_errors = errors[:,indexes, 0]
    #get average and std of vm_errors
    vm_errors = vm_errors.reshape(-1,1)
    #print the absolute average and standard deviation of errors for each feature only when mask is 1
    print(f'Absolute Average of Voltage Magnitude: {np.mean(np.abs(vm_errors))}')
    print(f'Absolute Standard Deviation of Voltage Magnitude: {np.std(np.abs(vm_errors))}')
    print("- "*40)

    indexes = np.where(masks[0, :, 1] == 1)[0]    
    va_errors = errors[:,indexes, 1]
    #get average and std of va_errors
    va_errors = va_errors.reshape(-1,1)
    #Print the absolute average and standard deviation of errors for each feature only when mask is 1
    print(f'Absolute Average of Voltage Angle: {np.mean(np.abs(va_errors))}')
    print(f'Absolute Standard Deviation of Voltage Angle: {np.std(np.abs(va_errors))}')
    print("- "*40)

    indexes = np.where(masks[0, :, 2] == 1)[0]
    ap_errors = errors[:,indexes, 2]
    #get average and std of ap_errors
    ap_errors = ap_errors.reshape(-1,1)
    #print the absolute average and standard deviation of errors for each feature only when mask is 1
    print(f'Absolute Average of Active Power: {np.mean(np.abs(ap_errors))}')
    print(f'Absolute Standard Deviation of Active Power: {np.std(np.abs(ap_errors))}')
    print("- "*40)

    indexes = np.where(masks[0, :, 3] == 1)[0]
    rp_errors = errors[:,indexes, 3]
    #get average and std of rp_errors
    rp_errors = rp_errors.reshape(-1,1)
    #print the absolute average and standard deviation of errors for each feature only when mask is 1
    print(f'Absolute Average of Reactive Power: {np.mean(np.abs(rp_errors))}')
    print(f'Absolute Standard Deviation of Reactive Power: {np.std(np.abs(rp_errors))}')
    print("- "*40)

    print(f'Average of all errors: {np.mean(errors)}')
    print(f'Standard Deviation of all errors: {np.std(errors)}')
    print("="*80)   
    # get indexes of loads and generators
    load_idx = np.where(types[0, :] == 2)[0]
    gen_idx = np.where(types[0, :] == 1)[0]

    # get error per feature
    # fig, axes = plt.subplots(2, 2, figsize=(22, 16))

    errors_loads = errors[:, load_idx, :]
    errors_gens = errors[:, gen_idx, :]

    error_per_feature = errors.reshape(-1, 4)
    error_per_feature_loads = errors_loads.reshape(-1, 4)
    error_per_feature_gens = errors_gens.reshape(-1, 4)

    if False:
        plt.style.use('seaborn-darkgrid')
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), tight_layout=True)
        for idx, ax in enumerate(axes.flatten()):

            N_all, bin_all, _ = ax.hist(error_per_feature[:, idx],
                                        bins=100,
                                        alpha=0.3,
                                        label='All Nodes',
                                        density=True)
            N_loads, bin_loads, _ = ax.hist(error_per_feature_loads[:, idx],
                                            bins=100,
                                            alpha=0.3,
                                            label='Load Nodes',
                                            density=True)
            N_gens, bin_gens, _ = ax.hist(error_per_feature_gens[:, idx],
                                          bins=100,
                                          alpha=0.3,
                                          label='Generator Nodes',
                                          density=True)

            ax.set_xlabel(f'{feature_names_output[idx]}')
            ax.set_ylabel('Probability')
            ax.legend()
            ax.set_yticks([])
            # max_N = max(max(N_all/N_all.sum()), max(N_loads/N_loads.sum()),
            #             max(N_gens/N_gens.sum()))
            # max_N_value = max(max(N_all), max(N_loads), max(N_gens))
            # ax.set_yticks([0,max_N_value], [0, max_N])
            # ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))

        plt.savefig('./results/'+case_name+'error_distribution_'+'.png')
        # plt.show()

    # plt.style.use('seaborn-darkgrid')

    print(errors.shape)
    n_nodes = errors.shape[1]
    n_bins = 300
    n_values_to_print_y = 7
    # Plot error per node average histogram
    # for n in range(errors.shape[0]):

    # error_per_node_loads = errors[n, load_idx, :].reshape(-1, 4)
    # error_per_node_gens = errors[n, gen_idx, :].reshape(-1, 4)
    for i in range(4):
        plot_counter += 1
        error_per_node_all = np.zeros((n_bins, n_nodes, 4))
        plt.subplot(3, 4, plot_counter)
        
        if plot_counter == 9:
            multiplier = 0.4
        elif plot_counter == 6:
            multiplier = 0.4
        elif plot_counter == 10:
            multiplier = 0.5
        elif i == 2:
            multiplier = 0.4
        elif i == 3:
            multiplier = 0.4
        else:
            multiplier = 0.8
        min_value = np.min(errors[:, :, i]) * multiplier
        max_value = np.max(errors[:, :, i]) * multiplier

        if abs(min_value) >= max_value:
            max_value = abs(min_value)
        elif abs(min_value) < max_value:
            min_value = -max_value

        print(f'min_value: {min_value}, max_value: {max_value}')

        bin_list = np.linspace(min_value, max_value, n_bins+1)
        bin_list_print = np.linspace(min_value, max_value, n_values_to_print_y)

        for n in range(n_nodes):
            hist, bins = np.histogram(errors[:, n, i],
                                      bins=bin_list,
                                      density=False)
            error_per_node_all[:, n, i] = hist/np.sum(hist)

        # indices = np.argsort(abs(error_per_node_all[:, :, i]).mean(axis=0))
        # print(indices.shape)
        # error_per_node_all = error_per_node_all[:, :, i]

        plt.imshow(error_per_node_all[:, :, i].T,
                   interpolation='nearest',
                   aspect='auto',
                   norm=mcolors.PowerNorm(0.3),
                   cmap='viridis',)

        # if i == 3:
        #     plt.colorbar(label='Probability')
        # else:
        #     plt.colorbar()

        if i == 0:
            plt.yticks(np.linspace(0, n_nodes, n_values_to_print_y)-0.5,
                       np.linspace(0, n_nodes, n_values_to_print_y, dtype=int))
        else:
            plt.yticks(np.linspace(0, n_nodes, n_values_to_print_y)-0.5,
                       [])

        if i == 0:
            plt.xticks(np.linspace(0, n_bins, n_values_to_print_y),
                       np.round(bin_list_print, 3),
                       rotation=45)
        else:
            plt.xticks(np.linspace(0, n_bins, n_values_to_print_y),
                       [f'{r:.1f}' for r in bin_list_print],
                       rotation=45)
        if i == 0:
            plt.ylabel(f'Case {case_name}\nNode Index', fontdict={'fontsize': 11})

        # plt.ylabel('Node Index')

        # if plot_counter > 8:
        #     plt.xlabel('Error')

        # plt.xlabel('Error')
        if plot_counter < 5:
            plt.title(f'{feature_names_output[i]}', fontdict={'fontsize': 12})


plt.subplots_adjust(bottom=0.145, right=0.98, top=0.95,
                    left=0.09, wspace=0.16, hspace=0.395) #hspace=0.326 wspawce=0.13
# cax = plt.axes([0.85, 0.1, 0.015, 0.9])
cax = plt.axes([0.09, 0.06, 0.88, 0.01])
plt.colorbar(location='bottom', cax=cax, label='Probability Density of the Error')

plt.savefig('./results/error_distribution_per_node.pdf',
            format='pdf', dpi=600)

plt.savefig('./results/error_distribution_per_node.eps',
            format='eps', dpi=600)

plt.show()
