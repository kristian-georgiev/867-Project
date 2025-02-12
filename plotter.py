import torch
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')
from collections import OrderedDict
from matplotlib.ticker import FormatStrFormatter
from sklearn.decomposition import PCA

from plotting_util import *

import pdb

def pca_directions(weights_accross_training):
    pca = PCA(n_components=2)
    pca.fit(weights_accross_training)
    dirs = pca.components_

    return dirs

def plot_loss_landscape(directions, 
                        test_dataset, 
                        support_dataset,
                        ml,
                        loss,
                        weights_over_time, 
                        shapes, 
                        state_dict_template,
                        plot_dir,
                        hparams):   

    gridsize = hparams.plot_gridsize

    # not doing that!
    # rescaling
    # dirs_norms = [get_rescaling_factors(d, shapes) for d in directions]
    # last_weights_norm = get_rescaling_factors(weights_over_time[-1], shapes)
    # for i in range(len(directions)):
    #     m = list(np.array(last_weights_norm) / np.array(dirs_norms[i]))
    #     directions[i] = multiply_filterwise(directions[i], shapes, m)
    # print("Rescaled directions!")


    offset = np.mean(weights_over_time, axis=0)

    w_coeffs = np.dot(weights_over_time - offset, directions.T)
    projected_ws = np.dot(w_coeffs, directions) + offset



    for i, weights in enumerate(weights_over_time):
        err = np.linalg.norm(projected_ws[i] - weights)
        print(f"For weight vector {i} in trajectory, we have a projection error of {err:.2f}, which is {err / np.linalg.norm(weights):.2f}%")
        # trajectory.append(projected_weights_coeffs)
    trajectory = projected_ws


    # constructs the test dataset
    X, Y = test_dataset
    X_s, Y_s = support_dataset
    # X_s = X_s[0]
    # Y_s = Y_s[0]
    print(f"Shape of test data is {X.shape}, and of test labels is {Y.shape}.")
    # X = X[0] # TODO update to an average over all test tasks
    # Y = Y[0]

    # trajectory = []

    fig, ax = plt.subplots()

    x_traj = [elt[0] for elt in trajectory]
    y_traj = [elt[1] for elt in trajectory]
    

    min_x, max_x = min(x_traj), max(x_traj)
    range_x = max_x - min_x
    margin_x = range_x * 0.1

    min_y, max_y = min(y_traj), max(y_traj)
    range_y = max_y - min_y
    margin_y = range_y * 0.1

    print("ranges for plotting", min_x, max_x, min_y, max_y)

    grid_x = np.linspace(min_x - margin_x, max_x + margin_x, gridsize)
    grid_y = np.linspace(min_y - margin_y, max_y + margin_y, gridsize)
    
    slow_w_loss_grid = np.empty((gridsize, gridsize))
    ft_loss_grid = np.empty((gridsize, gridsize))
    magn_grid = np.empty((gridsize, gridsize))
    vectors_grid_x = np.empty((gridsize, gridsize))
    vectors_grid_y = np.empty((gridsize, gridsize))
    accuracy = np.empty((gridsize, gridsize))
    for i in range(gridsize):
        for j in range(gridsize): 
            tup = loss_eval(grid_x[j],
                            grid_y[i],
                            offset,
                            loss,
                            directions,
                            X_s, Y_s,
                            X, Y,
                            ml,
                            shapes,
                            state_dict_template,
                            hparams)
            slow_w_loss_grid[i, j], ft_loss_grid[i, j], magn_grid[i, j], v, accuracy[i, j] = tup
            vectors_grid_x[i, j], vectors_grid_y[i, j] = v[0], v[1]

            print(f"At {i}, {j}, w\ coords {grid_x[j]}, {grid_y[i]}\
                the loss from slow weights is {slow_w_loss_grid[i, j]},\
                    the loss after fine-tuning is {ft_loss_grid[i, j]},\
                    projected directions are {vectors_grid_x[i, j]} \
                        and {vectors_grid_x[i, j], vectors_grid_y[i, j]}")

    # ft_loss_grid = np.clip(ft_loss_grid, 0, 20)

    vectors_grid_x /= range_x
    vectors_grid_y /= range_y

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
    # cbar.ax.set_yticklabels(["{:4.3f}".format(i) for i in cbar.ax.get_yticks()])
    # cbar.ax.set_major_formatter(FormatStrFormatter('%.2f'))
    print("Got contour plot!")
    print("LOSS GRID IS:")
    print(ft_loss_grid)

    # plots the trajectory
    plt.plot(x_traj, y_traj, c='black', lw='3')
    plt.scatter(x_traj[-1], y_traj[-1], marker='X', c='gold', s=300)
    
    plt.quiver(gx, gy, vectors_grid_x, vectors_grid_y)

    title = '_'.join([hparams.meta_learner, \
                         hparams.dataset, \
                         str(hparams.lr_finetune), \
                         str(hparams.n_inner_iter), \
                         str(hparams.index)])
    # ax.set_title(title)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    if hparams.fix_extractor: 
        addon = "fe_"
    elif hparams.fix_head: 
        addon = "fh_"
    else: 
        addon = ""
    plt.savefig(f"{plot_dir}/{addon}{title}.png", bbox_inches='tight')
    plt.close()


    fig, ax = plt.subplots()
    C = ax.contourf(gx, gy, accuracy,
                    levels=np.linspace(0,1.0,10),
                    cmap=plt.cm.Greens)
    print("Got contour plot!")
    print("ACC GRID IS:")
    print(accuracy)
    cbar = fig.colorbar(C)
    # plots the trajectory
    plt.plot(x_traj, y_traj, c='black', lw='3')
    plt.scatter(x_traj[-1], y_traj[-1], marker='X', c='gold', s=300)
    plt.quiver(gx, gy, vectors_grid_x, vectors_grid_y)
    # ax.set_title(title)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    if hparams.fix_extractor: 
        addon = "fe_"
    elif hparams.fix_head: 
        addon = "fh_"
    else: 
        addon = ""
    plt.savefig(f"{plot_dir}/{addon}acc_{title}.png", bbox_inches='tight')


    return title

def plot_progress(log, hparams):
    df = pd.DataFrame(log)

    fig, ax = plt.subplots(figsize=(6, 4))
    train_df = df[df['mode'] == 'train']
    test_df = df[df['mode'] == 'test']
    ax.plot(train_df['epoch'], train_df['acc'], label='Train')
    ax.plot(test_df['epoch'], test_df['acc'], label='Test')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    # ax.set_ylim(70, 100)
    fig.legend(ncol=2, loc='lower right')
    fig.tight_layout()
    title = '_'.join([hparams.meta_learner, \
                 hparams.dataset])
    ax.set_title(title)
    print(f'--- Plotting accuracy to {title}')
    fig.savefig(f"plots/train/{title}.png", bbox_inches='tight')
    plt.close(fig)


