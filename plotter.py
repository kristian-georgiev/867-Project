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
    pca.fit(weights_accross_training - last_w)
    dirs = pca.components_

    return dirs

def plot_loss_landscape(directions, 
                        test_dataset, 
                        ml,
                        loss,
                        weights_over_time, 
                        shapes, 
                        state_dict_template,
                        plot_dir,
                        hparams):   

    gridsize = hparams.plot_gridsize

    # rescaling
    dirs_norms = [get_rescaling_factors(d, shapes) for d in directions]
    last_weights_norm = get_rescaling_factors(weights_over_time[-1], shapes)
    for i in range(len(directions)):
        m = list(np.array(last_weights_norm) / np.array(dirs_norms[i]))
        directions[i] = multiply_filterwise(directions[i], shapes, m)
    print("Rescaled directions!")

    # constructs the test dataset
    X, Y = test_dataset
    print(f"Shape of test data is {X.shape}, and of test labels is {Y.shape}.")
    X = X[0] # TODO update to an average over all test tasks
    Y = Y[0]

    trajectory = []
    offset = weights_over_time[-1]

    for weights in weights_over_time:
        projected_weights = project_onto(weights, directions, offset)
        trajectory.append(projected_weights)
    trajectory = trajectory[1:]

    fig, ax = plt.subplots()

    x_traj = [elt[0] for elt in trajectory]
    y_traj = [elt[1] for elt in trajectory]
    

    min_x, max_x = min(x_traj), max(x_traj)
    margin_x = (max_x - min_x) * 0.1

    min_y, max_y = min(y_traj), max(y_traj)
    margin_y = (max_y - min_y) * 0.1

    grid_x = np.linspace(min_x - margin_x, max_x + margin_x, gridsize)
    grid_y = np.linspace(min_y - margin_y, max_y + margin_y, gridsize)
    
    slow_w_loss_grid = np.empty((gridsize, gridsize))
    ft_loss_grid = np.empty((gridsize, gridsize))
    magn_grid = np.empty((gridsize, gridsize))
    vectors_grid_x = np.empty((gridsize, gridsize))
    vectors_grid_y = np.empty((gridsize, gridsize))
    for i in range(gridsize):
        for j in range(gridsize): 
            tup = loss_eval(grid_x[j],
                            grid_y[i],
                            offset,
                            loss,
                            directions,
                            X, Y,
                            ml,
                            shapes,
                            state_dict_template,
                            hparams)
        print(f"At {i}, {j}, w\ coords {grid_x[j]}, {grid_y[i]}\
            the loss from slow weights is {tup[0]},\
                the loss after fine-tuning is {tup[1]},\
                projected directions are {tup[3]}")
        slow_w_loss_grid[i, j], ft_loss_grid[i, j], magn_grid[i, j], v = tup
        vectors_grid_x[i, j], vectors_grid_y[i, j] = v[0], v[1]

    # print("END LOSS IS:")
    # print(loss_eval(0, 0, offset, loss, directions, X, Y, ml, k_query, shapes, state_dict_template))

    # print("SLIGHT PERTUBATION OF IT IS:")
    # print(loss_eval(0.01, 0.01, offset, loss, directions, X, Y, ml, k_query, shapes, state_dict_template))

    # print("SWING OF IT IS:")
    # print(loss_eval(-0.09, 0.0025, offset, loss, directions, X, Y, ml, k_query, shapes, state_dict_template))

    gx, gy = np.meshgrid(grid_x, grid_y)
    C = ax.contourf(gx, gy, ft_loss_grid,
                    levels=gridsize,
                    cmap=plt.cm.coolwarm)
    cbar = fig.colorbar(C)
    print("Got contour plot!")
    print("LOSS GRID IS:")
    print(ft_loss_grid)

    # plots the trajectory
    plt.plot(x_traj, y_traj, c='black', lw='3')
    plt.scatter(x_traj[-1], y_traj[-1], marker='X', c='white', s=160)
    
    plt.quiver(gx, gy, vectors_grid_x, vectors_grid_y)

    filename = "trajectory.png"
    ax.set_title("Trajectory over training.")
    plt.savefig(f"{plot_dir}/{filename}")

    return filename

def plot_progress(log):
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


