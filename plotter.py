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

    print(flat_weight_np.shape)

    pca = PCA(n_components=2)
    pca.fit(flat_weight_np)
    dirs = pca.components_

    unflattened_dirs = unflatten(dirs, weights_accross_training[0])

    return unflattened_dirs

def plot_loss_landscape(directions, test_dataset, architecture, loss, k, weights_over_time):   
    k = 2
    # TODO: remove this 
    final_weights = weights_over_time[-1]
    # Does the rescaling
    for i in range(len(directions)): 
        for d, w in zip(directions[i].items(), final_weights.items()): 
            d[1] = d[1] * (np.linalg.norm(w[1].cpu().numpy()) / (np.linalg.norm(d[1]) + 1e-10))
            print(d[1])

    gridpoints = np.linspace(-1, 1, k)
    loss_grid = []

    for i, val_i in enumerate(gridpoints):
        loss_grid.append([])
        for j, val_j in enumerate(gridpoints):
            L = loss_eval(val_i, val_j , loss, directions, test_dataset, architecture)
            loss_grid[i].append(L)

    fig, ax = plt.subplots()
    C = ax.contour(gridpoints, gridpoints, loss_grid)
    ax.clabel(C)
    print("Got contour plot!")

    trajectory = []
    for weights in weights_over_time:
        projected_weights = project_onto(weights, directions)
        trajectory.append(projected_weights)

    x_traj = [elt[0] for elt in trajectory]
    y_traj = [elt[1] for elt in trajectory]

    plt.scatter(x_traj, y_traj)
    print(f"Trajectory is {trajectory}")

    filename = "trajectory.png"
    ax.set_title("Trajectory over training.")
    plt.show()
    plt.savefig(filename)

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


