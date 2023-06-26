from datetime import datetime
import os
import random

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from tqdm import tqdm

from datasets.PowerFlowData import PowerFlowData
from networks.MPN import MPN, MPN_simplenet
from utils.argument_parser import argument_parser
from utils.evaluation import evaluate_epoch
from utils.custom_loss_functions import Masked_L2_loss

import time
import pandapower as pp
import pickle


def load_cases(path):
    # load a pickle file containing the cases
    with open(path, 'rb') as f:
        cases = pickle.load(f)

    return cases


def load_net(sample, net, case_data, solution=None):
    # load the data into the pandapower network
    net.line['r_ohm_per_km'] = case_data[0]
    net.line['x_ohm_per_km'] = case_data[1]

    net.load['p_mw'] = case_data[4]
    net.load['q_mvar'] = case_data[5]

    net.gen['vm_pu'] = case_data[2]
    net.gen['p_mw'] = case_data[3]

    # instatiate the solver with solutions
    if solution is not None:
        # print(solution)
        net.res_bus['vm_pu'] = solution[:, 0]
        net.res_bus['va_degree'] = solution[:, 1]
        net.res_bus['p_mw'] = solution[:, 2]
        net.res_bus['q_mvar'] = solution[:, 3]

    return net


# Step 0: Parse Arguments and Setup
args = argument_parser()
models = {
    'MPN': MPN,
    'MPN_simplenet': MPN_simplenet,
}

# Training parameters
data_dir = args.data_dir
loss_fn = Masked_L2_loss(regularize=args.regularize,
                         regcoeff=args.regularization_coeff)
eval_loss_fn = Masked_L2_loss(regularize=False)

grid_case = "14"
net = pp.networks.case14()
model_path = "./models/testing/mpn_14.pt"
# model_path = "./models/model_20230623-3426.pt"

sample_number = 1000

# Network parameters
nfeature_dim = args.nfeature_dim
efeature_dim = args.efeature_dim
hidden_dim = args.hidden_dim
output_dim = args.output_dim
n_gnn_layers = args.n_gnn_layers
conv_K = args.K
dropout_rate = args.dropout_rate
model = models[args.model]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1234)
np.random.seed(1234)

# Step 1: Load data
testset = PowerFlowData(root=data_dir, case=grid_case,
                        split=[.5, .2, .3], task='test')
test_set_unnormalized = PowerFlowData(
    root=data_dir, case=grid_case, split=[.5, .2, .3], task='test', normalize=False)

test_set_mean = testset.xymean[0].to(device)
test_set_std = testset.xystd[0].to(device)

# Step 2: Load model
node_in_dim, node_out_dim, edge_dim = testset.get_data_dimensions()
assert node_in_dim == 16

model = model(
    nfeature_dim=nfeature_dim,
    efeature_dim=efeature_dim,
    output_dim=output_dim,
    hidden_dim=hidden_dim,
    n_gnn_layers=n_gnn_layers,
    K=conv_K,
    dropout_rate=dropout_rate
).to(device)

_to_load = torch.load(model_path)
model.load_state_dict(_to_load['model_state_dict'])

results = []

time_start_gnn = time.time()
for i, sample in enumerate(testset[:sample_number]):

    results.append(model(sample.to(device)))

time_end_gnn = time.time()

for i in range(len(results)):
    results[i] = results[i] * test_set_std + test_set_mean
    results[i] = results[i].detach().cpu().numpy()

cases = load_cases("./data/raw/case" + grid_case + "_reconstruction_case.pkl")


algorithms = ["nr", "iwamoto_nr",  "gs", "fdbx", "fdxb"]
algorithms = ["nr", "iwamoto_nr"]
algorithms = ["nr"]
results_nr = []
times_auto_init = []
loss_auto_init = 0
# Run the power flow with auto_init
for a in algorithms:
    print(f'Auto: Running {a}...')
    timer = 0

    for i, sample in enumerate(testset[:sample_number]):
        net = pp.networks.case14()
        net = load_net(sample, net, cases[i])
        t0 = time.time()
        pp.runpp(net, algorithm=a, init="auto", numba=False)
        t1 = time.time()
        # gt = sample.y * test_set_std + test_set_mean
        # loss_auto_init += eval_loss_fn(torch.tensor(net.res_bus.values), gt[:,:4], sample.x[:,10:14])
        result_pf = net.res_bus.values        
        result_pf = (torch.tensor(result_pf) - test_set_mean[:4]) / test_set_std[:4]
        results_nr.append(result_pf)
        # loss_auto_init += eval_loss_fn(result_pf, sample.y[:,:4], sample.x[:,10:14]).item()
        loss_auto_init += 0
        timer += t1 - t0

    times_auto_init.append(timer)

# Run the power flow with the results as initial values
times_result_init = []
loss_result_init = 0
for a in algorithms:
    print(f'Results: Running {a}...')
    timer = 0

    for i, sample in enumerate(testset[:sample_number]):
        net = pp.networks.case14()
        net = load_net(sample, net, cases[i], results[i])
        t0 = time.time()
        pp.runpp(net, algorithm=a, init="results", numba=False)
        t1 = time.time()

        result_pf = net.res_bus.values
        result_pf = (torch.tensor(result_pf) - test_set_mean[:4]) / test_set_std[:4]        
        loss_result_init += eval_loss_fn(result_pf, results_nr[i], sample.x[:,10:14]).item()
        timer += t1 - t0

    times_result_init.append(timer)

# Run the DC power flow
results_dc = []
times_dc = []
loss_dc = 0
for a in algorithms:
    print(f'DC: Running {a}...')
    timer = 0
    for i, sample in enumerate(testset[:sample_number]):
        net = pp.networks.case14()
        net = load_net(sample, net, cases[i], results[i])
        t0 = time.time()
        pp.rundcpp(net, algorithm=a, numba=False)
        t1 = time.time()
        results_dc.append(net.res_bus[["vm_pu", "va_degree", "p_mw", "q_mvar"]].values)
        results_dc[i] = (torch.tensor(results_dc[i]) - test_set_mean[:4]) / test_set_std[:4]
        # loss_dc += eval_loss_fn(results_dc[i], sample.y[:,:4], sample.x[:,10:14]).item()
        loss_dc += eval_loss_fn(results_dc[i], results_nr[i], sample.x[:,10:14]).item()
        # loss_dc += eval_loss_fn(torch.tensor(results_dc[i]), gt[:,:4], sample.x[:,10:14]).item()
        timer += t1 - t0
        # print(results_dc[i], sample.y[:,:4])

    times_dc.append(timer)

print("\n\n===========================================")
print("Results with auto_init:\n")

for a in algorithms:
    print(f"{a}: {times_auto_init[algorithms.index(a)]/sample_number}")
    print(f'Loss auto_init: {loss_auto_init/sample_number}')
print("-------------------------------------------")
print("GNNs: ", (time_end_gnn - time_start_gnn)/sample_number)
print("-------------------------------------------")
print("Results with results init: \n")
for a in algorithms:
    print(f"{a}: {times_result_init[algorithms.index(a)]/sample_number}")
    print(f'Loss result_init: {loss_result_init/sample_number}')
print("-------------------------------------------")
print("Results DC: \n")
for a in algorithms:
    print(f"{a}: {times_dc[algorithms.index(a)]/sample_number}")
    print(f'Loss DC: {loss_dc/sample_number}')
print("\n\n===========================================")
