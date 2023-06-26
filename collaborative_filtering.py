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


def collaborative_filtering_testing(y, mask, B, x_gt,f, eval_loss_fn=Masked_L2_loss(regularize=False)):

    # decision variables
    z_hat = cp.Variable((x_gt.shape[0], x_gt.shape[1]))

    distance = 1/2 * \
        cp.square(cp.pnorm(cp.multiply(y, mask)-cp.multiply(z_hat, mask), f))
    normalizer = cp.square(cp.pnorm(z_hat, f))
    # trace = cp.trace(cp.matmul(cp.multiply(z_hat,mask).T, cp.matmul(L, cp.multiply(z_hat,mask))))
    # trace = cp.trace(cp.matmul(z_hat.T, cp.matmul(L,z_hat)))
    trace = cp.norm(B@z_hat, 2)
    # lambda_z = 0.0001
    # prob = cp.Problem(cp.Minimize(1/2 * error + lambda_z*cp.norm(z_hat,1)))

    losses = []
    rnmse_list = []
    # for alpha in set  [0,10]  with 0,01 step
    alphas = np.arange(0, 5, 0.5)
    aplhas = []
    alphas = [0.1, 0.5, 1, 10]
    lambda_L_list = np.arange(0, 3, 0.5)
    lambda_z_list = np.arange(0, 3, 0.5)

    results = np.zeros((len(lambda_L_list), len(lambda_z_list)))

    for i, lambda_L in enumerate(lambda_L_list):
        for j, lambda_z in enumerate(lambda_z_list):
            prob = cp.Problem(cp.Minimize(
                distance + lambda_z*normalizer + lambda_L*trace))

            print("problem is solved...")
            prob.solve()
            print("status:", prob.status)
            # rnmse = np.sqrt(np.square(z_hat.value-x_gt).mean())/y.std()
            # rnmse_list.append(rnmse)
            # print(f"The rNMSE is: {np.round(rnmse,4)}")

            # print("z_hat: ", z_hat.value*mask.numpy())
            # print("x_gt: ", x_gt*mask.numpy())

            # print("mask: ", mask)

            z_tensor = torch.tensor(z_hat.value)
            x_gt_tensor = torch.tensor(x_gt)

            loss = eval_loss_fn(z_tensor, x_gt_tensor, mask)
            # print("loss: ", loss.item())
            # losses.append(loss.item())
            results[i, j] = loss.item()

    # plot a 2d heatmap of the results
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme()

    ax = sns.heatmap(results, annot=True, fmt=".2f", cmap="YlGnBu")
    ax.set_xlabel("lambda_z")
    ax.set_ylabel("lambda_L")

    plt.show()


def tikhonov_regularizer(alpha, L, y, mask):
    # Tikhonov regularization
    z_hat = np.matmul(np.matmul(np.linalg.inv(alpha*L + np.eye(L.shape[0])), L), y)        

    return z_hat

 
if __name__ == "__main__":
    data_dir = "./data/"
    grid_case = "5"
    grid_case = "14"
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
    # get laplacian matrix
    L = G.L.toarray()
    print(f'Laplacian: {L.shape}')

    # Get the data

    # x_gt is the actual values
    x_gt = trainset.y[:num_of_nodes, :4].numpy()
    print("x_gt: ", x_gt.shape, x_gt[0, :])

    # y is the observations, i.e. the features with missing values
    y = trainset.x[:num_of_nodes, 4:8]
    print("y: ", y.shape, y[0, :])

    # mask of values to be predicted
    mask = trainset.x[:num_of_nodes, 10:14]

    # find values x from ys
    # x_hat is the predicted values of the features with missing values

    # Create the problem
    rnmse_list = []
    print("problem is constructed...")
    f = x_gt.shape[1]
    print("f: ", f)

    collaborative_filtering_testing(y, mask, B,x_gt,f)

    eval_loss_fn=Masked_L2_loss(regularize=False)
    result = tikhonov_regularizer(1.25, L, y, mask)

    loss = eval_loss_fn(torch.tensor(result), torch.tensor(x_gt), mask)
    print("loss: ", loss.item())