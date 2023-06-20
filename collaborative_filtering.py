# import cvxopt as cvxopt
import cvxpy as cp
import numpy as np
import os
from torch_geometric.loader import DataLoader
from datasets.PowerFlowData import PowerFlowData
from utils.argument_parser import argument_parser
from utils.custom_loss_functions import Masked_L2_loss
from pygsp import graphs
import torch

data_dir = "./data/"
grid_case = "5"
# grid_case = "14"
# grid_case = "9"
# grid_case = "6470rte"
# grid_case = "118"

# Load the dataset
trainset = PowerFlowData(root=data_dir, case=grid_case,
                         split=[.5, .2, .3], task='train')
valset = PowerFlowData(root=data_dir, case=grid_case,
                       split=[.5, .2, .3], task='val')
testset = PowerFlowData(root=data_dir, case=grid_case,
                        split=[.5, .2, .3], task='test')
# train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False)
# test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

# Load adjacency matrix from file
file_path = data_dir + "/raw/case" + str(grid_case) + '_adjacency_matrix.npy'
adjacency_matrix = np.load(file_path)
print(adjacency_matrix.shape)

num_of_nodes = adjacency_matrix.shape[0]
print(f'Number of nodes: {num_of_nodes}')

# create graph from adjacency matrix
G = graphs.Graph(adjacency_matrix)

# get incidence matrix
G.compute_differential_operator()
B = G.D.toarray()
print(f'B: {B.shape}')
#get laplacian matrix
L = G.L.toarray()
print(f'Laplacian: {L.shape}')

eval_loss_fn = Masked_L2_loss(regularize=False)

# Get the data

# x_gt is the actual values
x_gt = trainset.y[:num_of_nodes, :4].numpy()
print("x_gt: ", x_gt.shape, x_gt[0, :])

# y is the observations, i.e. the features with missing values
y = trainset.x[:num_of_nodes, 4:8]
print("y: ", y.shape, y[0, :])

#mask of values to be predicted
mask = trainset.x[:num_of_nodes, 10:14]

# find values x from ys
# x_hat is the predicted values of the features with missing values

# Create the problem
rnmse_list = []
print("problem is constructed...")
f = x_gt.shape[1]
print("f: ", f)

# decision variables
z_hat = cp.Variable((x_gt.shape[0], x_gt.shape[1]))

# x_hat = cp.Variable((x_gt.shape[0],x_gt.shape[1]))
error = cp.square(cp.pnorm(cp.multiply(y,mask)-cp.multiply(z_hat,mask), f))
regularizer = cp.norm(z_hat, 1)

# lambda_z = 0.0001
# prob = cp.Problem(cp.Minimize(1/2 * error + lambda_z*cp.norm(z_hat,1)))

losses = []
rnmse_list = []
#for alpha in set  [0,10]  with 0,01 step
alphas = np.arange(0, 5, 0.5)
aplhas = []
alphas = [0.1,1,10]
for alpha in alphas:

    prob = cp.Problem(cp.Minimize(error + alpha*cp.sum_squares(B@(cp.multiply(z_hat,mask)))))

    print("problem is solved...")
    prob.solve()
    print("status:", prob.status)
    rnmse = np.sqrt(np.square(z_hat.value-x_gt).mean())/y.std()
    rnmse_list.append(rnmse)
    print(f"The rNMSE is: {np.round(rnmse,4)}")

    # print("z_hat: ", z_hat.value*mask.numpy())
    # print("x_gt: ", x_gt*mask.numpy())

    # print("mask: ", mask)

    z_tensor = torch.tensor(z_hat.value)
    x_gt_tensor = torch.tensor(x_gt)

    loss = eval_loss_fn(z_tensor, x_gt_tensor, mask)
    # print("loss: ", loss.item())
    losses.append(loss.item())

#plot losses and rmse in different subplots
import matplotlib.pyplot as plt
fig, axs = plt.subplots(2)
axs[0].plot(alphas, losses)
axs[0].set_title('Loss')
axs[1].plot(alphas, rnmse_list)
axs[1].set_title('rNMSE')
plt.show()

