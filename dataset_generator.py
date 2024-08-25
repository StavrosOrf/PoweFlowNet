"""Synthetic Power Flow Data Generator with Pandapower
Format: 
    - edge_features: [num_samples, num_edges, 7]
    - node_features: [num_samples, num_nodes, 6]
        - index: index of the node, starting from 0
        - type: 1 for generator, 2 for load
"""
import time
import argparse
import pandas as pd
import pandapower as pp
import numpy as np
import networkx as nx
import multiprocessing as mp
import os

from utils.data_utils import perturb_topology

number_of_samples = 30000
number_of_processes = 10
ENFORCE_Q_LIMS = False

def create_case3():
    net = pp.create_empty_network()
    net.sn_mva = 100
    b0 = pp.create_bus(net, vn_kv=345., name='bus 0')
    b1 = pp.create_bus(net, vn_kv=345., name='bus 1')
    b2 = pp.create_bus(net, vn_kv=345., name='bus 2')
    pp.create_ext_grid(net, bus=b0, vm_pu=1.02, name="Grid Connection")
    pp.create_load(net, bus=b2, p_mw=10.3, q_mvar=3, name="Load")
    # pp.create_gen(net, bus=b1, p_mw=0.5, vm_pu=1.03, name="Gen", max_p_mw=1)
    pp.create_line(net, from_bus=b0, to_bus=b1, length_km=10, name='line 01', std_type='NAYY 4x50 SE')
    pp.create_line(net, from_bus=b1, to_bus=b2, length_km=5, name='line 01', std_type='NAYY 4x50 SE')
    pp.create_line(net, from_bus=b2, to_bus=b0, length_km=20, name='line 01', std_type='NAYY 4x50 SE')
    
    net.line['c_nf_per_km'] = pd.Series(0., index=net.line['c_nf_per_km'].index, name=net.line['c_nf_per_km'].name)
    
    return net

def remove_c_nf(net):
    net.line['c_nf_per_km'] = pd.Series(0., index=net.line['c_nf_per_km'].index, name=net.line['c_nf_per_km'].name)
    
def unify_vn(net):
    for node_id in range(net.bus['vn_kv'].shape[0]):
        net.bus['vn_kv'][node_id] = max(net.bus['vn_kv'])

def get_trafo_z_pu(net):
    # for trafo_id in net.trafo.index:
    #     # net.trafo['i0_percent'][trafo_id] = 0.
    #     # net.trafo['pfe_kw'][trafo_id] = 0.
    #     net.trafo[trafo_id, 'i0_percent'] = 0.
    #     net.trafo[trafo_id, 'pfe_kw'] = 0.
        
    net.trafo.loc[net.trafo.index, 'i0_percent'] = 0.
    net.trafo.loc[net.trafo.index, 'pfe_kw'] = 0.
    
    z_pu = net.trafo['vk_percent'].values / 100. * 1000. / net.sn_mva
    r_pu = net.trafo['vkr_percent'].values / 100. * 1000. / net.sn_mva
    x_pu = np.sqrt(z_pu**2 - r_pu**2)
    
    return x_pu, r_pu
    
def get_line_z_pu(net):
    r = net.line['r_ohm_per_km'].values * net.line['length_km'].values
    x = net.line['x_ohm_per_km'].values * net.line['length_km'].values
    from_bus = net.line['from_bus']
    to_bus = net.line['to_bus']
    vn_kv_to = net.bus['vn_kv'][to_bus].to_numpy()
    # vn_kv_to = pd.Series(vn_kv_to)
    zn = vn_kv_to**2 / net.sn_mva
    r_pu = r/zn
    x_pu = x/zn
    
    return r_pu, x_pu

def get_adjacency_matrix(net):
    multi_graph = pp.topology.create_nxgraph(net)
    A = nx.adjacency_matrix(multi_graph).todense() 
    
    return A

def generate_data(sublist_size, rng, base_net_create, num_lines_to_remove=0, num_lines_to_add=0):
    edge_features_list = []
    node_features_list = []
    # graph_feature_list = []

    while len(edge_features_list) < sublist_size:
        net = base_net_create()
        remove_c_nf(net)
        
        success_flag, net = perturb_topology(net, num_lines_to_remove=num_lines_to_remove, num_lines_to_add=num_lines_to_add) # TODO 
        if success_flag == 1:
            exit()
        n = net.bus.values.shape[0]
        A = get_adjacency_matrix(net)
        
        net.bus['name'] = net.bus.index

        r = net.line['r_ohm_per_km'].values    
        x = net.line['x_ohm_per_km'].values
        # c = net.line['c_nf_per_km'].values
        le = net.line['length_km'].values
        # x = case['branch'][:, 3]
        # b = case['branch'][:, 4]
        # tau = case['branch'][:, 8]  # ratio

        Pg = net.gen['p_mw'].values
        # Pmin = 
        Pd = net.load['p_mw'].values
        Qd = net.load['q_mvar'].values

        # rng = np.random.default_rng()
        r = rng.uniform(0.8*r, 1.2*r, r.shape[0])
        _x_min = np.where(x>=0, 0.8*x, 1.2*x) # in 6470rte, line reactance might be negative
        _x_max = np.where(x>=0, 1.2*x, 0.8*x)
        x = rng.uniform(_x_min, _x_max, x.shape[0])
        # c = np.random.uniform(0.8*c, 1.2*c, c.shape[0])
        le = rng.uniform(0.8*le, 1.2*le, le.shape[0])
        
        # tau = np.random.uniform(0.8*tau, 1.2*tau, case['branch'].shape[0])
        # angle = np.random.uniform(-0.2, 0.2, case['branch'].shape[0])
    
        Vg = rng.uniform(1.00, 1.05, net.gen['vm_pu'].shape[0])
        Pg = rng.normal(Pg, 0.1*np.abs(Pg), net.gen['p_mw'].shape[0])
        
        # Pd = np.random.uniform(0.5*Pd, 1.5*Pd, net.load['p_mw'].shape[0])
        Pd = rng.normal(Pd, 0.1*np.abs(Pd), net.load['p_mw'].shape[0])
        # Qd = np.random.uniform(0.5*Qd, 1.5*Qd, net.load['q_mvar'].shape[0])
        Qd = rng.normal(Qd, 0.1*np.abs(Qd), net.load['q_mvar'].shape[0])
        
        net.line['r_ohm_per_km'] = r 
        net.line['x_ohm_per_km'] = x 

        net.gen['vm_pu'] = Vg
        net.gen['p_mw'] = Pg

        net.load['p_mw'] = Pd
        net.load['q_mvar'] = Qd

        try:
            net['converged'] = False
            pp.runpp(net, algorithm='nr', init="results", numba=False, enforce_q_lims=ENFORCE_Q_LIMS)
        except:
            if not net['converged']:
                # print(f"net['converged'] = {net['converged']}")
                print(f'Failed to converge, current sample number: {len(edge_features_list)}')
                import pandapower as pp
                continue        

        # Graph feature
        # baseMVA = x[0]['baseMVA']

        # Create a vector od branch features including start and end nodes,r,x,b,tau,angle
        edge_features = np.zeros((net.line.shape[0], 4))
        edge_features[:, 0] = net.line['from_bus'].values
        edge_features[:, 1] = net.line['to_bus'].values
        edge_features[:, 2], edge_features[:, 3] = get_line_z_pu(net)
        
        trafo_edge_features = np.zeros((net.trafo.shape[0], 4))
        trafo_edge_features[:, 0] = net.trafo['hv_bus'].values
        trafo_edge_features[:, 1] = net.trafo['lv_bus'].values
        trafo_edge_features[:, 2], trafo_edge_features[:, 3] = get_trafo_z_pu(net)
        
        edge_features = np.concatenate((edge_features, trafo_edge_features), axis=0)

        # Record node features
        #   bus type: 0 - slack bus, 1 - generator, 2 - load
        types = np.ones(n)*2 # type = load
        for j in range(net.gen.shape[0]):    
            # find index of case['gen'][j,0] in case['bus'][:,0]
            index = np.where(net.gen['bus'].values[j] == net.bus['name'])[0][0] 
            if ENFORCE_Q_LIMS:
                if net.res_gen['q_mvar'][j] <= net.gen['min_q_mvar'][j] + 1e-6 \
                    or net.res_gen['q_mvar'][j] >= net.gen['max_q_mvar'][j] - 1e-6:
                        continue # seen as load bus
            types[index] = 1  # type = generator
        for j in range(net.ext_grid.shape[0]):
            index = np.where(net.ext_grid['bus'].values[j] == net.bus['name'])[0][0]
            types[index] = 0 # type = slack bus
        for j in range(net.load.shape[0]):    
            index = np.where(net.load['bus'].values[j] == net.bus['name'])[0][0]
            pass
        
        #   Create a vector of node features including index, type, Vm, Va, Pd, Qd, Gs, Bs    
        node_features = np.zeros((n, 6))
        node_features[:, 0] = net.bus['name'].values # index
        node_features[:, 1] = types  # type
        # Vm ----This changes for Load Buses
        # if net.res_bus['vm_pu'].shape[0] == 0:
        #     pass
        node_features[:, 2] = net.res_bus['vm_pu']  # Vm
        # Va ----This changes for every bus excecpt slack bus
        node_features[:, 3] = net.res_bus['va_degree']  # Va
        node_features[:, 4] = net.res_bus['p_mw'] / net.sn_mva    # P / pu
        node_features[:, 5] = net.res_bus['q_mvar'] / net.sn_mva  # Q / pu
        # node_features_y[:, 6] = case['bus'][:, 4]  # Gs
        # node_features_y[:, 7] = case['bus'][:, 5]  # Bs

        edge_features_list.append(edge_features)
        node_features_list.append(node_features)
        # graph_feature_list.append(baseMVA)

        if len(edge_features_list) % 10 == 0 or len(edge_features_list) == sublist_size:
            print(f'[Process {os.getpid()}] Current sample number: {len(edge_features_list)}')
            
    return edge_features_list, node_features_list

def generate_data_parallel(num_samples, num_processes, base_net_create, num_lines_to_remove=0, num_lines_to_add=0):
    sublist_size = num_samples // num_processes
    parent_rng = np.random.default_rng(123456)
    streams = parent_rng.spawn(num_processes)
    pool = mp.Pool(processes=num_processes)
    args = [[sublist_size, st, base_net_create, num_lines_to_remove, num_lines_to_add] for st in streams]
    results = pool.starmap(generate_data, args)
    # results = generate_data(*args[0]) # DEBUG LINE
    pool.close()
    pool.join()
    
    edge_features_list = []
    node_features_list = []
    for sub_res in results:
        edge_features_list += sub_res[0]
        node_features_list += sub_res[1]
        
    return edge_features_list, node_features_list

if __name__ == '__main__':
    # arguments
    parser = argparse.ArgumentParser(prog='Power Flow Data Generator', description='')
    parser.add_argument('--case', type=str, default='118', help='e.g. 118, 14, 6470rte')
    parser.add_argument('--num_lines_to_remove', '-r', type=int, default=0, help='Number of lines to remove')
    parser.add_argument('--num_lines_to_add', '-a', type=int, default=0, help='Number of lines to add')
    args = parser.parse_args()

    num_lines_to_remove = args.num_lines_to_remove
    num_lines_to_add = args.num_lines_to_add
    case = args.case

    if case == '3':
        base_net_create = create_case3
    elif case == '14':
        base_net_create = pp.networks.case14
    elif case == '118':
        base_net_create = pp.networks.case118
    elif case == '6470rte':
        base_net_create = pp.networks.case6470rte
    else:
        print('Invalid test case.')
        exit()
    if num_lines_to_remove > 0 or num_lines_to_add > 0:
        complete_case_name = 'case' + case + 'perturbed' + f'{num_lines_to_remove:1d}' + 'r' + f'{num_lines_to_add:1d}' + 'a'
    else:
        complete_case_name = 'case' + case
    base_net = base_net_create()
    base_net.bus['name'] = base_net.bus.index
    print(base_net.bus)
    print(base_net.line)
    
    # Generate data
    edge_features_list, node_features_list = generate_data_parallel(number_of_samples, number_of_processes, base_net_create,
                                                                    num_lines_to_remove=num_lines_to_remove, num_lines_to_add=num_lines_to_add)
    
    # Turn the lists into numpy arrays
    edge_features = np.array(edge_features_list)
    node_features = np.array(node_features_list)
    # graph_features = np.array(graph_feature_list)

    # Print the shapes
    print(f'edge_features shape: {edge_features.shape}')
    print(f'node_features_x shape: {node_features.shape}')

    print(f'range of edge_features "from": {np.min(edge_features[:,:,0])} - {np.max(edge_features[:,:,0])}')
    print(f'range of edge_features "to": {np.min(edge_features[:,:,1])} - {np.max(edge_features[:,:,1])}')

    print(f'range of node_features "index": {np.min(node_features[:,:,0])} - {np.max(node_features[:,:,0])}')

    # print(f"A. {A}")
    # print(f"edge_features. {edge_features}")
    # print(f"node_features_x. {node_features_x}")
    # print(f"node_features_y. {node_features_y}")

    # save the features
    os.makedirs("./data/raw", exist_ok=True)
    with open("./data/raw/"+complete_case_name+"_edge_features.npy", 'wb') as f:
        np.save(f, edge_features)

    with open("./data/raw/"+complete_case_name+"_node_features.npy", 'wb') as f:
        np.save(f, node_features)

    # with open("./data/"+test_case+"_graph_features.npy", 'wb') as f:
    #     np.save(f, graph_features)

    # with open("./data/raw/"+test_case+"_adjacency_matrix.npy", 'wb') as f:
    #     np.save(f, A)