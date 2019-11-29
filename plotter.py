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
    pca = PCA(n_components=2)
    last_w = weights_accross_training[-1]
    # pca.fit(weights_accross_training[0:len(weights_accross_training) - 1])
    pca.fit(weights_accross_training - last_w)
    dirs = pca.components_

    return dirs

def plot_loss_landscape(directions, 
                        test_dataset, 
                        architecture, 
                        loss, 
                        k, 
                        weights_over_time, 
                        shapes, 
                        state_dict_template,
                        plot_dir):   

    # # does rescaling
    # for i in range(len(directions)):
    #     for key, old_val in weights_over_time[-1].items():
    #         if key in weights_over_time[-1].keys():
    #             if isinstance(weights_over_time[-1][key], torch.Tensor):
    #                 directions[i][key] *= (np.linalg.norm(weights_over_time[-1][key].cpu().numpy())\
    #                  / (np.linalg.norm(directions[i][key]) + 1e-10))

    # rescaling
    dirs_norms = [get_rescaling_factors(d, shapes) for d in directions]
    last_weights_norm = get_rescaling_factors(weights_over_time[-1], shapes)
    for i in range(len(directions)):
        m = list(np.array(last_weights_norm) / np.array(dirs_norms[i]))
        directions[i] = multiply_filterwise(directions[i], shapes, m)


    print("^^^^^^^^^^^^^^^^")
    print(directions[:,0:5])
    print("Rescaled directions!")
    # constructs the test dataset
    X, Y = test_dataset
    print(X.shape, Y.shape)
    X = X[0]
    Y = Y[0]
    # X = X[20,10,:,:,:]
    # Y = Y[20,:]
    # print(X.shape, Y.shape)
    # x, y, z = X.shape[2:]
    # X = X.permute(2, 3, 4, 0, 1).reshape(x, y, z, -1).permute(3, 0, 1, 2)
    # Y = Y.reshape(-1)

    trajectory = []
    offset = weights_over_time[-1]

    for weights in weights_over_time:
        projected_weights = project_onto(weights, directions, offset)
        trajectory.append(projected_weights)
    trajectory = trajectory[1:]

    fig, ax = plt.subplots()

    x_traj = [elt[0] for elt in trajectory]
    y_traj = [elt[1] for elt in trajectory]
    

    min_x = min(x_traj)
    max_x = max(x_traj)
    scale_x = (max_x - min_x) * 0.1
    min_y = min(y_traj)
    max_y = max(y_traj)
    scale_y = (max_y - min_y) * 0.1
    print(f"X MIN AND MAX ARE {min_x}, {max_x}")
    print(f"Y MIN AND MAX ARE {min_y}, {max_y}")

    grid_x = np.linspace(min_x - scale_x, max_x + scale_x, k)
    # print(grid_x)
    # grid_x = np.append(grid_x, x_traj)
    # print(grid_x)
    # print("above are grid_x old and new size!!!!")
    # gridpoints_x = grid_x.tolist() * k
    # print(f"There are {len(gridpoints_x)} X gridpoints and k is {k}.")

    grid_y = np.linspace(min_y - scale_y, max_y + scale_y, k)
    # grid_y = np.append(grid_y, y_traj)
    
    # gridpoints_y = []
    # for p in grid_y: 
    #     gridpoints_y.extend([p] * k)
    
    k = len(grid_x)

    loss_grid = np.empty((k, k))
    for i in range(k):
        for j in range(k): 
            loss_grid[i, j] = loss_eval(grid_x[j],
                                        grid_y[i],
                                        offset,
                                        loss,
                                        directions,
                                        X, Y,
                                        architecture,
                                        shapes,
                                        state_dict_template)
            # print(f"i {grid_x[i]}, \
            # j {grid_y[j]}, loss {loss_grid[i, j]}")

    # print(grid_x)
    # print(grid_y)
    # print(loss_grid)

    print("END LOSS IS:")
    print(loss_eval(0, 0, offset, loss, directions, X, Y, architecture, shapes, state_dict_template))

    print("SLIGHT PERTUBATION OF IT IS:")
    print(loss_eval(0.01, 0.01, offset, loss, directions, X, Y, architecture, shapes, state_dict_template))


    print("SWING OF IT IS:")
    print(loss_eval(-0.09, 0.0025, offset, loss, directions, X, Y, architecture, shapes, state_dict_template))


    gx, gy = np.meshgrid(grid_x, grid_y)
    C = ax.contourf(gx, gy, loss_grid)# , levels=5) #, cmap=plt.cm.coolwarm)
    cbar = fig.colorbar(C)
    print("Got contour plot!")
    print("LOSS GRID IS:")
    print(loss_grid)
    print("MESHGRID IS:")
    # print(gx, gy)

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


