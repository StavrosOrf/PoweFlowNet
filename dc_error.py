import time
import pandapower as pp
import numpy as np
import pickle
from utils.custom_loss_functions import Masked_L2_loss
import torch
from datasets.PowerFlowData import PowerFlowData
# write file documentation here




# dict_keys(['bus', 'load', 'sgen', 'motor', 'asymmetric_load', 'asymmetric_sgen', 'storage', 'gen', 'switch', 'shunt', 'svc', 'ext_grid', 'line', 'trafo', 'trafo3w', 'impedance', 'tcsc', 'dcline', 'ward', 'xward', 'measurement', 'pwl_cost', 'poly_cost', 'characteristic', 'controller', 'group', 'line_geodata', 'bus_geodata', '_empty_res_bus', '_empty_res_ext_grid', '_empty_res_line', '_empty_res_trafo', '_empty_res_load', '_empty_res_asymmetric_load', '_empty_res_asymmetric_sgen', '_empty_res_motor', '_empty_res_sgen', '_empty_res_shunt', '_empty_res_svc', '_empty_res_switch', '_empty_res_impedance', '_empty_res_tcsc', '_empty_res_dcline', '_empty_res_ward', '_empty_res_xward', '_empty_res_trafo_3ph', '_empty_res_trafo3w', '_empty_res_bus_3ph', '_empty_res_ext_grid_3ph', '_empty_res_line_3ph', '_empty_res_asymmetric_load_3ph', '_empty_res_asymmetric_sgen_3ph', '_empty_res_storage', '_empty_res_storage_3ph', '_empty_res_gen',
        #   '_ppc', '_ppc0', '_ppc1', '_ppc2', '_is_elements', '_pd2ppc_lookups', 'version', 'format_version', 'converged', 'OPF_converged', 'name', 'f_hz', 'sn_mva', '_empty_res_load_3ph', '_empty_res_sgen_3ph', 'std_types', 'res_bus', 'res_line', 'res_trafo', 'res_trafo3w', 'res_impedance', 'res_ext_grid', 'res_load', 'res_motor', 'res_sgen', 'res_storage', 'res_shunt', 'res_gen', 'res_ward', 'res_xward', 'res_dcline', 'res_asymmetric_load', 'res_asymmetric_sgen', 'res_switch', 'res_tcsc', 'res_svc', 'res_bus_est', 'res_line_est', 'res_trafo_est', 'res_trafo3w_est', 'res_impedance_est', 'res_switch_est', 'res_bus_sc', 'res_line_sc', 'res_trafo_sc', 'res_trafo3w_sc', 'res_ext_grid_sc', 'res_gen_sc', 'res_sgen_sc', 'res_switch_sc', 'res_bus_3ph', 'res_line_3ph', 'res_trafo_3ph', 'res_ext_grid_3ph', 'res_shunt_3ph', 'res_load_3ph', 'res_sgen_3ph', 'res_storage_3ph', 'res_asymmetric_load_3ph', 'res_asymmetric_sgen_3ph', 'user_pf_options'])

# net = pp.networks.GBnetwork()

# algorithm (str, “nr”) - algorithm that is used to solve the power flow problem.

# The following algorithms are available:

# “nr” Newton-Raphson (pypower implementation with numba accelerations)

# “iwamoto_nr” Newton-Raphson with Iwamoto multiplier (maybe slower than NR but more robust)

# “bfsw” backward/forward sweep (specially suited for radial and weakly-meshed networks)

# “gs” gauss-seidel (pypower implementation)

# “fdbx” fast-decoupled (pypower implementation)

# “fdxb” fast-decoupled (pypower implementation)

# print(net)
# print(net.keys())


cases = [pp.networks.case14, pp.networks.case118, pp.networks.case6470rte]
# cases = [pp.networks.case14]
case_names = ['14', '118', '6470rte']

counter = 0
number_of_samples = 1000

dc_losses = {"14": [], "118": [], "6470rte": []}

eval_loss_fn = Masked_L2_loss(regularize=False)

for i, base_net in enumerate(cases):
    current_sample_number = 0
    testset = PowerFlowData(root="./data/", case=case_names[i], split=[.5, .2, .3], task='test')
    
    mask = testset[0].x[:,10:14]   
    
    mask[:,0] = 0
    mask[:,3] = 0
    # mask[:,2] = 0
    print(f'Mask: {mask}')
    test_set_mean = testset.xymean[0]
    test_set_std = testset.xystd[0]
    
    while True:


        net = base_net()


        r = net.line['r_ohm_per_km'].values    
        x = net.line['x_ohm_per_km'].values
        


        Pg = net.gen['p_mw'].values        
        Pd = net.load['p_mw'].values
        Qd = net.load['q_mvar'].values

        r = np.random.uniform(0.8*r, 1.2*r, r.shape[0])
        x = np.random.uniform(0.8*x, 1.2*x, x.shape[0])                
        
        Vg = np.random.uniform(1.00, 1.05, net.gen['vm_pu'].shape[0])
        Pg = np.random.normal(Pg, 0.1*np.abs(Pg), net.gen['p_mw'].shape[0])
                
        Pd = np.random.normal(Pd, 0.1*np.abs(Pd), net.load['p_mw'].shape[0])        
        Qd = np.random.normal(Qd, 0.1*np.abs(Qd), net.load['q_mvar'].shape[0])
        
        net.line['r_ohm_per_km'] = r
        net.line['x_ohm_per_km'] = x

        net.gen['vm_pu'] = Vg
        net.gen['p_mw'] = Pg

        net.load['p_mw'] = Pd
        net.load['q_mvar'] = Qd


        try:    
            pp.runpp(net, algorithm='nr', init="auto", numba=False)
        except:
            print(f'Failed to converge, current sample number: {len(current_sample_number)}')        
            continue        
        
        denormalized_result_nr = net.res_bus.values        
            # print(result_pf)

        # result_nr = torch.tensor(denormalized_result_nr)
        result_nr = (torch.tensor(denormalized_result_nr) - test_set_mean[:4]) / test_set_std[:4]
        
        # print(f'Results NR: \n{result_nr}')        

        net = base_net()
        net.line['r_ohm_per_km'] = r
        net.line['x_ohm_per_km'] = x

        net.gen['vm_pu'] = Vg
        net.gen['p_mw'] = Pg

        net.load['p_mw'] = Pd
        net.load['q_mvar'] = Qd

        pp.rundcpp(net, calculate_voltage_angles=True, numba=False)

        results_dc = net.res_bus.values
        # for k in range(results_dc.shape[0]): 
        #     results_dc[k][3] = denormalized_result_nr[k][3]          

        # results_dc = torch.tensor(results_dc)
        results_dc = (torch.tensor(results_dc) - test_set_mean[:4]) / test_set_std[:4]
        # print(f'DC Results: \n{results_dc}')      
        
        loss_dc = eval_loss_fn(results_dc, result_nr, mask).item()

        dc_losses[case_names[i]].append(loss_dc)

        print(f'loss: {loss_dc}')
        current_sample_number += 1


        if current_sample_number >= number_of_samples:
            break

    print(f'Case {case_names[i]} done')

#print statistics
print(f'Average losses: {np.mean(dc_losses["14"])} {np.mean(dc_losses["118"])} {np.mean(dc_losses["6470rte"])}')
print(f'Std losses: {np.std(dc_losses["14"])} {np.std(dc_losses["118"])} {np.std(dc_losses["6470rte"])}')
print(f'Max losses: {np.max(dc_losses["14"])} {np.max(dc_losses["118"])} {np.max(dc_losses["6470rte"])}')
print(f'Min losses: {np.min(dc_losses["14"])} {np.min(dc_losses["118"])} {np.min(dc_losses["6470rte"])}')
print(f'Median losses: {np.median(dc_losses["14"])} {np.median(dc_losses["118"])} {np.median(dc_losses["6470rte"])}')
print(f'25th percentile losses: {np.percentile(dc_losses["14"], 25)} {np.percentile(dc_losses["118"], 25)} {np.percentile(dc_losses["6470rte"], 25)}')
print(f'75th percentile losses: {np.percentile(dc_losses["14"], 75)} {np.percentile(dc_losses["118"], 75)} {np.percentile(dc_losses["6470rte"], 75)}')
print(f'95th percentile losses: {np.percentile(dc_losses["14"], 95)} {np.percentile(dc_losses["118"], 95)} {np.percentile(dc_losses["6470rte"], 95)}')
print(f'99th percentile losses: {np.percentile(dc_losses["14"], 99)} {np.percentile(dc_losses["118"], 99)} {np.percentile(dc_losses["6470rte"], 99)}')
#print average loss


