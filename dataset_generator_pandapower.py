import time
import pandapower as pp
import numpy as np
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

number_of_samples = 10

test_case = 'case5'
# base_net = pp.networks.case6470rte()
base_net = pp.networks.case5()
base_net.bus['name'] = base_net.bus.index
print(base_net.bus)
print(base_net.line)
print(base_net.gen)

# Get Adjacency Matrix
bus_names = base_net.bus['name'].values.tolist()
n = base_net.bus.values.shape[0]
A = np.zeros((n, n))
for edge1,edge2 in base_net.line[['from_bus', 'to_bus']].values:
    
    edge_1 = bus_names.index(edge1)
    edge_2 = bus_names.index(edge2)

    A[edge_1, edge_2] = 1
    A[edge_2, edge_1] = 1

edge_features_list = []
node_features_x_list = []
node_features_y_list = []
graph_feature_list = []

ref_bus = base_net.ext_grid['bus'].values[0]
print(f'ref_bus: {ref_bus}')

while True:
    net = base_net
    # net = pp.networks.case6470rte()
    # net = pp.networks.case5()
    net.bus['name'] = base_net.bus.index

    r = net.line['r_ohm_per_km'].values    
    x = net.line['x_ohm_per_km'].values
    # c = net.line['c_nf_per_km'].values
    le = net.line['length_km'].values
    # x = case['branch'][:, 3]
    # b = case['branch'][:, 4]
    # tau = case['branch'][:, 8]  # ratio

    Pmax = net.gen['max_p_mw'].values
    Pmin = net.gen['min_p_mw'].values
    Pmw = net.gen['p_mw'].values

    Pd = net.load['p_mw'].values
    Qd = net.load['q_mvar'].values

    # r = np.random.uniform(0.8*r, 1.2*r, r.shape[0])    
    # x = np.random.uniform(0.8*x, 1.2*x, x.shape[0])
    # le = np.random.uniform(0.8*le, 1.2*le, le.shape[0])
   
    vg = np.random.uniform(0.95, 1.05, net.gen['vm_pu'].shape[0])
    # Pg = np.random.uniform(0.25*Pmax, 0.75*Pmax, net.gen['p_mw'].shape[0])
    Pg = np.random.uniform(0.25*Pmw, 1.75*Pmw, net.gen['p_mw'].shape[0])
    
    
    Pd = np.random.uniform(0.5*Pd, 1.5*Pd, net.load['p_mw'].shape[0])
    Qd = np.random.uniform(0.5*Qd, 1.5*Qd, net.load['q_mvar'].shape[0])
    
    net.line['r_ohm_per_km'] = r
    net.line['x_ohm_per_km'] = x

    net.gen['vm_pu'] = vg
    net.gen['p_mw'] = Pg

    net.load['p_mw'] = Pd
    net.load['q_mvar'] = Qd

    try:    
        pp.runpp(net, algorithm='nr', init="results", numba=False)
    except:
        print(f'Failed to converge, current sample number: {len(edge_features_list)}')        
        continue        

    # Graph feature
    # baseMVA = x[0]['baseMVA']

    # Create a vector od branch features including start and end nodes,r,x,b,tau,angle
    edge_features = np.zeros((net.line.shape[0], 7))
    edge_features[:, 0] = net.line['from_bus'].values + 1
    edge_features[:, 1] = net.line['to_bus'].values + 1
    edge_features[:, 2] = net.line['r_ohm_per_km'].values * net.line['length_km'].values
    edge_features[:, 3] = net.line['x_ohm_per_km'].values * net.line['length_km'].values
    edge_features[:, 4] = 0
    edge_features[:, 5] = 0
    edge_features[:, 6] = 0

    # Create a vector of node features including index, type, Vm, Va, Pd, Qd, Gs, Bs, Pg
    # case['bus'] = x[0]['bus']

    node_features_x = np.zeros((n, 9))
    node_features_x[:, 0] = net.bus['name'].values + 1# index
    # Va ----This changes for every bus excecpt slack bus
    node_features_x[:, 3] = np.zeros((n, )) #Va

    node_features_x[ref_bus, 3] = 0 #Va for reference bus
    
    # node_features_x[:, 6] = np.zeros((n,1)) # Gs
    # node_features_x[:, 7] = np.zeros((n,1)) # Bs
    # Vm is 1 if type is not "generator" else it is case['gen'][:,j]
    vm = np.ones(n)
    types = np.ones(n)*2
    for j in range(net.gen.shape[0]):    
        # find index of case['gen'][j,0] in case['bus'][:,0]
        index = np.where(net.gen['bus'].values[j] == net.bus['name'])[0][0]        
        vm[index] = net.gen['vm_pu'].values[j]  # Vm = Vg
        types[index] = 1  # type = generator
        node_features_x[index, 8] = net.gen['p_mw'].values[j]  # Pg
    
    types[ref_bus] = 3 # 
    node_features_x[:, 2] = vm  # Vm
    node_features_x[:, 1] = types  # type
    
    for j in range(net.load.shape[0]):    
        # find index of case['gen'][j,0] in case['bus'][:,0]
        index = np.where(net.load['bus'].values[j] == net.bus['name'])[0][0]        
        node_features_x[index, 4] = Pd[j]  # Pd
        node_features_x[index, 5] = Qd[j] # Qd

    # Create a vector of node features including index, type, Vm, Va, Pd, Qd, Gs, Bs    
    node_features_y = np.zeros((n, 8))
    node_features_y[:, 0] = net.bus['name'].values + 1 # index
    node_features_y[:, 1] = types  # type
    # Vm ----This changes for Load Buses
    node_features_y[:, 2] = net.res_bus['vm_pu']  # Vm
    # Va ----This changes for every bus excecpt slack bus
    node_features_y[:, 3] = net.res_bus['va_degree']  # Va
    node_features_y[:, 4] = net.res_bus['p_mw']  # P
    node_features_y[:, 5] = net.res_bus['q_mvar']  # Q
    # node_features_y[:, 6] = case['bus'][:, 4]  # Gs
    # node_features_y[:, 7] = case['bus'][:, 5]  # Bs

    edge_features_list.append(edge_features)
    node_features_x_list.append(node_features_x)
    node_features_y_list.append(node_features_y)
    # graph_feature_list.append(baseMVA)

    if len(edge_features_list) == number_of_samples:
        break
    elif len(edge_features_list) % 100 == 0:
        print(f'Current sample number: {len(edge_features_list)}')

    if len(edge_features_list) % 1000 == 0:
        # Turn the lists into numpy arrays
        edge_features = np.array(edge_features_list)
        node_features_x = np.array(node_features_x_list)
        node_features_y = np.array(node_features_y_list)
        # graph_features = np.array(graph_feature_list)

        with open("./data/raw/"+test_case+"_edge_features.npy", 'wb') as f:
            np.save(f, edge_features)

        with open("./data/raw/"+test_case+"_node_features_x.npy", 'wb') as f:
            np.save(f, node_features_x)

        with open("./data/raw/"+test_case+"_node_features_y.npy", 'wb') as f:
            np.save(f, node_features_y)

        # with open("./data/"+test_case+"_graph_features.npy", 'wb') as f:
        #     np.save(f, graph_features)

    with open("./data/raw/"+test_case+"_adjacency_matrix.npy", 'wb') as f:
        np.save(f, A)

edge_features = np.array(edge_features_list)
node_features_x = np.array(node_features_x_list)
node_features_y = np.array(node_features_y_list)
# graph_features = np.array(graph_feature_list)

# Print the shapes
print(f'Adjacency matrix shape: {A.shape}')
print(f'edge_features shape: {edge_features.shape}')
print(f'node_features_x shape: {node_features_x.shape}')
print(f'node_features_y shape: {node_features_y.shape}')
# print(f'graph_features shape: {graph_features.shape}')

print(f'range of edge_features "from": {np.min(edge_features[:,:,0])} - {np.max(edge_features[:,:,0])}')
print(f'range of edge_features "to": {np.min(edge_features[:,:,1])} - {np.max(edge_features[:,:,1])}')

print(f'range of node_features_x "index": {np.min(node_features_x[:,:,0])} - {np.max(node_features_x[:,:,0])}')

print(f'range of node_features_y "index": {np.min(node_features_y[:,:,0])} - {np.max(node_features_y[:,:,0])}')

exit()
#  Computation time experimental comparison beginning (will be moved to other file later on)

# calculate power flow for every algorithm and calculate time
algorithms = ["nr", "iwamoto_nr",  "gs", "fdbx", "fdxb"]
times = []

for a in algorithms:
    t0 = time.time()
    # pp.runpp(net, algorithm=a)
    pp.runpp(net, algorithm=a, init="results", numba=False)
    t1 = time.time()
    times.append(t1 - t0)

for a in algorithms:
    print(f"{a}: {times[algorithms.index(a)]}")


# print(net.res_bus.vm_pu)
# print(net.res_line.loading_percent)

# calculate power flow for every algorithm and calculate time 1000 times
# algorithms = ["nr", "iwamoto_nr",  "gs", "fdbx", "fdxb"]
algorithms = ["nr", "iwamoto_nr", "fdbx", "fdxb"]
times = []

for a in algorithms:
    print(a)
    t0 = time.time()
    for i in range(1000):
        pp.runpp(net, algorithm=a, init="auto", numba=False)
    t1 = time.time()
    times.append(t1 - t0)

for a in algorithms:
    print(f"{a}: {times[algorithms.index(a)]/1000}")