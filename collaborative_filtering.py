# import cvxopt as cvxopt
import cvxpy as cp
import numpy as np
import os
from torch_geometric.loader import DataLoader
from datasets.PowerFlowData import PowerFlowData
from utils.argument_parser import argument_parser
from utils.custom_loss_functions import Masked_L2_loss
from pygsp import graphs

data_dir = "./data/"
grid_case = 5

#Load the dataset
trainset = PowerFlowData(root=data_dir, case=grid_case, split=[.5, .2, .3], task='train')
valset = PowerFlowData(root=data_dir, case=grid_case, split=[.5, .2, .3], task='val')
testset = PowerFlowData(root=data_dir, case=grid_case, split=[.5, .2, .3], task='test')
# train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False)
# test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

#Load adjacency matrix from file
file_path = data_dir + "/raw/case" + str(grid_case) + '_adjacency_matrix.npy'
adjacency_matrix = np.load(file_path)
print(adjacency_matrix.shape)

#create graph from adjacency matrix
G = graphs.Graph(adjacency_matrix)

#get incidence matrix
G.compute_differential_operator()
B = G.D.toarray()

eval_loss_fn = Masked_L2_loss(regularize=False)

#Get the data
x = trainset.x[:,4:8]
print(x.shape, x[0])
y = trainset.y[:,:4]
print(y.shape, y[0])

rnmse_list = []
print("problem is constructed...")
x_hat = cp.Variable((x.shape[0],x.shape[1]))
error = cp.square(cp.norm((y-x_hat),2))
regularizer = cp.norm(B@x_hat,1)

alpha = 0.5
prob = cp.Problem(cp.Minimize(error + alpha*cp.sum_squares(B@x_hat)))
print("problem is solved...")
prob.solve()
print("status:", prob.status)
rnmse = np.sqrt(np.square(x_hat.value-x).mean())/y.std()
print(f"The rNMSE is: {np.round(rnmse,4)}")    

