import torch
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')
from collections import OrderedDict
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
    return 0
    
def project_onto(weights, directions):
    """Projects list of weights on list of
    orthogonal directions
    
    Arguments:
        weights {list(OrderedDict())} -- weights at some epoch
        directions {list(OrderedDict())} -- list of 2 orthogonal directions
        we got from PCA
    
    Returns:
        projection coeffs -- tuple of two coefficients 
        of the projected weights
    """
    projection_coeffs = []

    flat_weights = flatten(weights)

    for dir in directions:
        coeff = np.dot(flat_weights, dir) / np.linalg.norm(dir)
        projection_coeffs.append(coeff)

    return projection_coeffs

