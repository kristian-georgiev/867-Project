import torch
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')
from collections import OrderedDict
from sklearn.decomposition import PCA

from plotting_util import *

import pdb

def pca_directions(weights_accross_training):
    weights_accross_training = cumsum(weights_accross_training)
    if isinstance(weights_accross_training[0], np.ndarray):
        flat_weight_list = [flatten(weights) for weights in weights_accross_training]
    else:
        flat_weight_list = [flatten(weights).cpu().numpy() for weights in weights_accross_training]

    print("Flattened weights.")

    flat_weight_np = np.array(flat_weight_list)


    pca = PCA(n_components=2)
    pca.fit(flat_weight_np)
    dirs = pca.components_

    unflattened_dirs = unflatten(dirs, weights_accross_training[0])

    return unflattened_dirs

def plot_loss_landscape(directions, test_dataset, architecture, loss, k, weights_over_time, plot_dir):   

    # does rescaling
    for i in range(len(directions)):
        for key, old_val in weights_over_time[-1].items():
            if key in weights_over_time[-1].keys():
                if isinstance(weights_over_time[-1][key], torch.Tensor):
                    directions[i][key] *= (np.linalg.norm(weights_over_time[-1][key].cpu().numpy())\
                     / (np.linalg.norm(directions[i][key]) + 1e-10))

    # constructs the test dataset
    X, Y = test_dataset
    print(X.shape, Y.shape)
    X = X[2]
    Y = Y[2]
    # X = X[20,10,:,:,:]
    # Y = Y[20,:]
    # print(X.shape, Y.shape)
    # x, y, z = X.shape[2:]
    # X = X.permute(2, 3, 4, 0, 1).reshape(x, y, z, -1).permute(3, 0, 1, 2)
    # Y = Y.reshape(-1)

    trajectory = []
    for weights in weights_over_time:
        projected_weights = project_onto(weights, directions)
        trajectory.append(projected_weights)
    trajectory = trajectory[1:]

    fig, ax = plt.subplots()

    x_traj = [elt[0] for elt in trajectory]
    y_traj = [elt[1] for elt in trajectory]
    

    min_x = min(x_traj)
    max_x = max(x_traj)
    scale_x = abs(min_x - max_x) * 0.1
    min_y = min(y_traj)
    max_y = max(y_traj)
    scale_y = abs(min_y - max_y) * 0.1

    grid_x = np.linspace(min_x - scale_x, max_x + scale_x, k)
    gridpoints_x = grid_x.tolist() * k

    grid_y = np.linspace(min_y - scale_y, max_y + scale_y, k)
    gridpoints_y = []
    for p in grid_y: 
        gridpoints_y.extend([p] * k)
    
    theta_star = weights_over_time[-1] # final weights

    # def wrapper(val_i, val_j): 
    #     return loss_eval(val_i, val_j, theta_star, loss, directions, X, Y, architecture)

    loss_grid = np.empty((k, k))
    for i in range(k):
        for j in range(k): 
            loss_grid[i,j] = loss_eval(grid_y[i], grid_x[j], theta_star, loss, directions, X, Y, architecture)

    # loss_grid = map(wrapper, gridpoints_y, gridpoints_x)
    # loss_grid = np.reshape(np.array(list(loss_grid)), (k, k))

    print(loss_grid)
    C = ax.contourf(grid_x, grid_y, loss_grid, levels=20, cmap=plt.cm.coolwarm)
    cbar = fig.colorbar(C)
    print("Got contour plot!")

    # plots the trajectory
    # plt.scatter(x_traj, y_traj)
    plt.plot(x_traj, y_traj, c='black', lw='3')
    plt.scatter(x_traj[-1], y_traj[-1], marker='X', c='white', s=160)
    

    filename = "trajectory.png"
    ax.set_title("Trajectory over training.")
    plt.savefig(f"{plot_dir}/{filename}")

    return filename

def plot_progress(log):
    # Generally you should pull your plotting code out of your training
    # script but we are doing it here for brevity.
    df = pd.DataFrame(log)

    fig, ax = plt.subplots(figsize=(6, 4))
    train_df = df[df['mode'] == 'train']
    test_df = df[df['mode'] == 'test']
    ax.plot(train_df['epoch'], train_df['acc'], label='Train')
    ax.plot(test_df['epoch'], test_df['acc'], label='Test')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(70, 100)
    fig.legend(ncol=2, loc='lower right')
    fig.tight_layout()
    fname = 'maml-accs.png'
    print(f'--- Plotting accuracy to {fname}')
    fig.savefig(fname)
    plt.close(fig)


