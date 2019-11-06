import torch
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')
from collections import OrderedDict
import torch.nn.functional as F
from sklearn.decomposition import PCA


def flatten(weights_dict):
    flat_weights = [weights_dict[t].reshape(-1) for t in weights_dict]
    return torch.cat(flat_weights)

def unflatten(dirs, weights_desired_shape):
    assert (len(flatten(weights_desired_shape)) == len(dirs[0])),\
         f"We need dimensions {len(dirs)} of dirs and \
             {len(flatten(weights_desired_shape))} of state_dict to match up"

    unflattened_dirs = []

    for d in dirs:
        unflattened_dir = OrderedDict()
        
        for w in weights_desired_shape:
            l = len(weights_desired_shape[w].reshape(-1))
            shape = weights_desired_shape[w].shape
            unflattened_dir[w] = d[ :l].reshape(shape)
            d = d[l: ]
        unflattened_dirs.append(unflattened_dir)
        
    return unflattened_dirs

def loss_eval(i, j , loss, directions, test_dataset, architecture):
    # TODO: implement this f-n
    weights = {key: i*directions[0][key] + j*directions[1][key] for key in directions[0].keys()}
    old_state = architecture.state_dict()

    new_state = {}
    for key, old_val in old_state.items():
        if key in weights.keys():
            new_state[key] = weights[key]
        else:
            new_state[key] = old_val

    architecture.load_state_dict(new_state)

    X_test, Y_test = test_dataset
    Y_pred = architecture.forward(X_test)
    loss_val = loss(Y_pred, Y_test)

    return loss_val
    
def project_onto(weights, directions):
    """Projects list of weights on list of
    orthogonal directions
    
    Arguments:
        weights {list(OrderedDict())} -- weights at some epoch
        directions {list(OrderedDict())} -- list of 2 orthogonal directions
        we got from PCA
    
    Returns:
        projected weights -- a list of the tuples of two coefficients 
        of the projected weights
    """
    # TODO: implement this f-n
    return weights

