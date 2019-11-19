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
    if isinstance(weights_accross_training[0], np.ndarray):
        flat_weight_list = [flatten(weights) for weights in weights_accross_training]
    else:
        flat_weight_list = [flatten(weights).cpu().numpy() for weights in weights_accross_training]

    print("Flattened weights.")

    flat_weight_np = np.array(flat_weight_list)
    final_weights = np.sum(flat_weight_np, axis=0)
    gradient_updates = flat_weight_np[2:] # remove init and 0 vector

    pca_input = np.vstack((gradient_updates, final_weights))

    pca = PCA(n_components=2)
    pca.fit(pca_input)
    dirs = pca.components_

    unflattened_dirs = unflatten(dirs, weights_accross_training[0])

    return unflattened_dirs

def plot_loss_landscape(directions, test_dataset, architecture, loss, k, weights_over_time, plot_dir):   
    final_weights = weights_over_time[-50]

    # does rescaling
    # for i in range(len(directions)):
    #     for key, old_val in final_weights.items():
    #         if key in final_weights.keys():
    #             if isinstance(final_weights[key], torch.Tensor):
    #                 directions[i][key] *= (np.linalg.norm(final_weights[key].cpu().numpy()) / (np.linalg.norm(directions[i][key]) + 1e-10))

    # constructs the test dataset
    X, Y = test_dataset
    x, y, z = X.shape[2:]
    X = X.permute(2, 3, 4, 0, 1).reshape(x, y, z, -1).permute(3, 0, 1, 2)
    Y = Y.reshape(-1)

    trajectory = []
    for weights in weights_over_time:
        projected_weights = project_onto(weights, directions)
        trajectory.append(projected_weights)

    fig, ax = plt.subplots()

    x_traj = [elt[0] for elt in trajectory]
    y_traj = [elt[1] for elt in trajectory]
    

    min_x = min(x_traj)
    max_x = max(x_traj)
    scale_x = abs(min_x - max_x) * 0.5
    min_y = min(y_traj)
    max_y = max(y_traj)
    scale_y = abs(min_y - max_y) * 0.5

    grid_x = np.linspace(min_x - scale_x, max_x + scale_x, k)
    gridpoints_x = grid_x.tolist() * k

    grid_y = np.linspace(min_y - scale_y, max_y + scale_y, k)
    gridpoints_y = []
    for p in grid_y: 
        gridpoints_y.extend([p] * k)

    def wrapper(val_i, val_j): 
        return loss_eval(val_i, val_j , loss, directions, X, Y, architecture)
    loss_grid = map(wrapper, gridpoints_y, gridpoints_x)
    loss_grid = np.reshape(np.array(list(loss_grid)), (k,k))

    print(loss_grid)
    C = ax.contour(grid_x, grid_y, loss_grid)
    ax.clabel(C)
    print("Got contour plot!")

    plt.scatter(x_traj, y_traj)
    print(f"Trajectory is {trajectory}")

    # plt.scatter(x_traj, y_traj)

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


