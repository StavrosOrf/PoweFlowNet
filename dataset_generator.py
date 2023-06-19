# write an example code for a power flow using pypower

from pygsp import graphs, plotting
import pypower.api as pp
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm


def print_Bus_data(case, case2):
    print("\n======================================")
    print("Bus Data")
    print("======================================")
    print("%7s %8s %7s %7s %7s %7s %7s %7s %7s %7s %7s %7s %7s" % ('bus', 'type',
                                                                   'Pd', 'Qd', 'Gs', 'Bs', 'area', 'Vm', 'Va', 'baseKV', 'zone', 'maxVm', 'minVm'))
    print("======================================"*3)
    for i in range(n):
        print("%7d %8d %7.1f %7.1f %7.1f %7.1f %7d %7.3f %7.3f %7.1f %7d %7.3f %7.3f" % (case['bus'][i, 0], case['bus'][i, 1], case['bus'][i, 2], case['bus'][i, 3], case['bus'][
            i, 4], case['bus'][i, 5], case['bus'][i, 6], case['bus'][i, 7], case['bus'][i, 8], case['bus'][i, 9], case['bus'][i, 10], case['bus'][i, 11], case['bus'][i, 12]))

    print("======================================"*3)
    for i in range(n):
        print("%7d %8d %7.1f %7.1f %7.1f %7.1f %7d %7.3f %7.3f %7.1f %7d %7.3f %7.3f" % (case2['bus'][i, 0], case2['bus'][i, 1], case2['bus'][i, 2], case2['bus'][i, 3], case2['bus'][
            i, 4], case2['bus'][i, 5], case2['bus'][i, 6], case2['bus'][i, 7], case2['bus'][i, 8], case2['bus'][i, 9], case2['bus'][i, 10], case2['bus'][i, 11], case2['bus'][i, 12]))


def print_Gen_data(case, case2):
    # print generator data with column names
    print("\n======================================")
    print("Generator Data")
    print("======================================")

    print("%7s %7s %7s %7s %7s %7s %7s %7s %7s %7s" % ('bus', 'Pg',
                                                       'Qg', 'Qmax', 'Qmin', 'Vg', 'mBase', 'status', 'Pmax', 'Pmin'))
    print("======================================"*3)
    for i in range(case['gen'].shape[0]):
        print("%7d %7.1f %7.1f %7.1f %7.1f %7.3f %7.1f %7d %7.1f %7.1f" % (case['gen'][i, 0], case['gen'][i, 1], case['gen'][i, 2], case[
            'gen'][i, 3], case['gen'][i, 4], case['gen'][i, 5], case['gen'][i, 6], case['gen'][i, 7], case['gen'][i, 8], case['gen'][i, 9]))
    print("======================================"*3)

    for i in range(case2['gen'].shape[0]):
        print("%7d %7.1f %7.1f %7.1f %7.1f %7.3f %7.1f %7d %7.1f %7.1f" % (case2['gen'][i, 0], case2['gen'][i, 1], case2['gen'][i, 2], case2[
            'gen'][i, 3], case2['gen'][i, 4], case2['gen'][i, 5], case2['gen'][i, 6], case2['gen'][i, 7], case2['gen'][i, 8], case2['gen'][i, 9]))


def print_Branch_data(case, case2):
    # print branch data with column names
    #
    print("\n======================================")
    print("Branch Data")
    print("======================================")
    print("%7s %7s %7s %7s %7s %7s %7s %7s %7s %7s" % ('fbus', 'tbus',
                                                       'r', 'x', 'b', 'rateA', 'rateB', 'rateC', 'ratio', 'angle'))
    print("======================================"*3)
    for i in range(case['branch'].shape[0]):
        print("%7d %7d %7.3f %7.3f %7.5f %7.1f %7.1f %7.1f %7.3f %7.3f" % (case['branch'][i, 0], case['branch'][i, 1], case['branch'][i, 2], case['branch']
                                                                           [i, 3], case['branch'][i, 4], case['branch'][i, 5], case['branch'][i, 6], case['branch'][i, 7], case['branch'][i, 8], case['branch'][i, 9]))

    print("======================================"*3)
    for i in range(case2['branch'].shape[0]):
        print("%7d %7d %7.3f %7.3f %7.5f %7.1f %7.1f %7.1f %7.3f %7.3f" % (case2['branch'][i, 0], case2['branch'][i, 1], case2['branch'][i, 2], case2['branch'][
            i, 3], case2['branch'][i, 4], case2['branch'][i, 5], case2['branch'][i, 6], case2['branch'][i, 7], case2['branch'][i, 8], case2['branch'][i, 9]))

def get_Admittance_matrices(case):

    # Get admittance matrices using pypower
    ppc = pp.ext2int(case)
    baseMVA, bus, gen, branch = ppc["baseMVA"], ppc["bus"], ppc["gen"], ppc["branch"]
    Ybus, Yf, Yt = pp.makeYbus(baseMVA, bus, branch)
    #Builds the vector of complex bus power injections.
    Sbus = pp.makeSbus(baseMVA, bus, gen)

    print(Ybus.shape)
    print(Yf.shape)
    print(Yt.shape)

    return Ybus, Yf, Yt, Sbus

# Dataset generation parameters

number_of_samples = 100000
#Tested case files: case4gs, case14, case30, case118
test_case = 'case14'
base_case = pp.case14()

PD_factor = 1

# Get Adjacency Matrix
bus_names = base_case['bus'][:, 0].tolist()
n = base_case['bus'].shape[0]
A = np.zeros((n, n))
for edge in base_case['branch']:

    edge_1 = bus_names.index(edge[0])
    edge_2 = bus_names.index(edge[1])

    A[edge_1, edge_2] = 1
    A[edge_2, edge_1] = 1

edge_features_list = []
node_features_x_list = []
node_features_y_list = []
graph_feature_list = []

while True:
    case = base_case

    # Get random values for the parameters
    r = case['branch'][:, 2]
    x = case['branch'][:, 3]
    b = case['branch'][:, 4]
    tau = case['branch'][:, 8]  # ratio

    Pmax = case['gen'][:, 8]
    Pmin = case['gen'][:, 9]
    Pd = case['bus'][:, 2]

    r = np.random.uniform(0.8*r, 1.2*r, case['branch'].shape[0])
    x = np.random.uniform(0.8*x, 1.2*x, case['branch'].shape[0])
    b = np.random.uniform(0.1*b, 2.0*b, case['branch'].shape[0])
    # tau = np.random.uniform(0.8*tau, 1.2*tau, case['branch'].shape[0]) # NOTE shouldn't change this. but does it make a difference?
    # angle = np.random.uniform(-0.2, 0.2, case['branch'].shape[0])     # NOTE should theoretically not matter

    vg = np.random.uniform(0.95, 1.05, case['gen'].shape[0])
    Pg = np.random.uniform(0.25*Pmax, 0.75*Pmax, case['gen'].shape[0])

    Pd = np.random.uniform(0.5*Pd, 1.5*Pd, case['bus'].shape[0])
    Qd = np.random.uniform(0.5*Pd, 1.5*Pd, case['bus'].shape[0])
    
    # case['branch'][:, 2] = r    
    # case['branch'][:, 3] = x
    case['branch'][:, 4] = b
    # case['branch'][:, 8] = tau
    # case['branch'][:, 9] = angle
    case['branch'][:, 8] = 0.
    case['branch'][:, 9] = 0.

    # # print(vg)
    # case['gen'][:, 5] = vg
    # # print(case['gen'][:, 5])
    # # print(Pg)
    # case['gen'][:, 1] = Pg
    # # print(case['gen'][:, 1])

    # case['bus'][:, 2] = Pd * PD_factor
    # case['bus'][:, 3] = Qd

    # print_Bus_data(base_case, case)
    # print_Gen_data(base_case, case)
    # print_Branch_data(base_case, case)

    ppopt = pp.ppoption()
    ppopt["PF_MAX_IT"] = 10
    ppopt['VERBOSE'] = True
    x = pp.runpf(case,ppopt=ppopt)

    if x[1] == 0:
        print(f'Failed to converge, current sample number: {len(edge_features_list)}')
        continue

    # Graph feature
    baseMVA = x[0]['baseMVA']

    # Create a vector od branch features including start and end nodes,r,x,b,tau,angle
    edge_features = np.zeros((case['branch'].shape[0], 7))
    edge_features[:, 0] = case['branch'][:, 0]
    edge_features[:, 1] = case['branch'][:, 1]
    edge_features[:, 2] = case['branch'][:, 2]
    edge_features[:, 3] = case['branch'][:, 3]
    edge_features[:, 4] = case['branch'][:, 4]
    edge_features[:, 5] = case['branch'][:, 8]
    edge_features[:, 6] = case['branch'][:, 9]

    # Create a vector of node features including index, type, Vm, Va, Pd, Qd, Gs, Bs, Pg
    case['bus'] = x[0]['bus']

    node_features_x = np.zeros((case['bus'].shape[0], 9))
    node_features_x[:, 0] = case['bus'][:, 0]  # index
    node_features_x[:, 1] = case['bus'][:, 1]  # type
    # Va ----This changes for every bus excecpt slack bus
    node_features_x[:, 3] = np.zeros(case['bus'].shape[0])
    node_features_x[:, 4] = case['bus'][:, 2]  # Pd
    node_features_x[:, 5] = case['bus'][:, 3]  # Qd
    node_features_x[:, 6] = case['bus'][:, 4]  # Gs
    node_features_x[:, 7] = case['bus'][:, 5]  # Bs
    # Vm is 1 if type is not "generator" else it is case['gen'][:,j]
    vm = np.ones(case['bus'].shape[0])
    for j in range(case['gen'].shape[0]):
        # find index of case['gen'][j,0] in case['bus'][:,0]
        index = np.where(case['bus'][:, 0] == case['gen'][j, 0])[0][0]        
        vm[index] = case['gen'][j, 5]  # Vm = Vg
        node_features_x[index, 8] = case['gen'][j, 1]  # Pg

    node_features_x[:, 2] = vm  # Vm

    # Create a vector of node features including index, type, Vm, Va, Pd, Qd, Gs, Bs
    case['bus'] = x[0]['bus']
    node_features_y = np.zeros((case['bus'].shape[0], 8))
    node_features_y[:, 0] = case['bus'][:, 0]  # index
    node_features_y[:, 1] = case['bus'][:, 1]  # type
    # Vm ----This changes for Load Buses
    node_features_y[:, 2] = case['bus'][:, 7]
    # Va ----This changes for every bus excecpt slack bus
    node_features_y[:, 3] = case['bus'][:, 8]
    node_features_y[:, 4] = case['bus'][:, 2]  # Pd
    node_features_y[:, 5] = case['bus'][:, 3]  # Qd
    node_features_y[:, 6] = case['bus'][:, 4]  # Gs
    node_features_y[:, 7] = case['bus'][:, 5]  # Bs

    edge_features_list.append(edge_features)
    node_features_x_list.append(node_features_x)
    node_features_y_list.append(node_features_y)
    graph_feature_list.append(baseMVA)

    if len(edge_features_list) == number_of_samples:
        break

# Turn the lists into numpy arrays
edge_features = np.array(edge_features_list)
node_features_x = np.array(node_features_x_list)
node_features_y = np.array(node_features_y_list)
graph_features = np.array(graph_feature_list)

# Print the shapes
print(f'Adjacency matrix shape: {A.shape}')
print(f'edge_features shape: {edge_features.shape}')
print(f'node_features_x shape: {node_features_x.shape}')
print(f'node_features_y shape: {node_features_y.shape}')
print(f'graph_features shape: {graph_features.shape}')

# save the features
with open("./data/"+test_case+"_edge_features.npy", 'wb') as f:
    np.save(f, edge_features)

with open("./data/"+test_case+"_node_features_x.npy", 'wb') as f:
    np.save(f, node_features_x)

with open("./data/"+test_case+"_node_features_y.npy", 'wb') as f:
    np.save(f, node_features_y)

with open("./data/"+test_case+"_graph_features.npy", 'wb') as f:
    np.save(f, graph_features)

with open("./data/"+test_case+"_adjacency_matrix.npy", 'wb') as f:
    np.save(f, A)
