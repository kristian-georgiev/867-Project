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

import pdb

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

def loss_eval(i, j , theta_star, loss, directions, X, Y, architecture):
    """Evaluate loss on test set at the point given by i, j, directions
    
    Arguments:
        i {float} -- coeff. for first direction, in [-1, 1]
        j {float} -- coeff. for second direction, in [-1, 1]
        theta_star -- offset of the affine subspace spanned by the directions
        loss {function} -- loss f-n to be evaluated
        directions {list(OrderedDict)} -- list of two directions we got from PCA
        test_dataset {tuple} -- tuple of (X, Y) loaded from dataloader
        architecture {torch.Sequential} -- PyTorch net
    
    Returns:
        float -- loss evaluated at 
        architecture(weights = i * dir[0] + j * dir[1]) on test_dataset
    """
    weights = {key: theta_star[key].numpy() + i * directions[0][key] + j * directions[1][key] \
        for key in directions[0].keys()}

    old_state = architecture.state_dict()
    new_state = OrderedDict()

    # update keys that are present in weights
    # for context, net.state_dict() contains tensors that are
    # neither weights, nor biases, and they are not present in the 
    # weights ordered dictionary
    for key, old_val in old_state.items():
        if key in weights.keys():
            if isinstance(weights[key], torch.Tensor):
                new_state[key] = weights[key]
            else:
                new_state[key] = torch.from_numpy(weights[key])
        else:
            new_state[key] = old_val

    architecture.load_state_dict(new_state)
    architecture.eval()
    
    with torch.no_grad(): 
        Y_pred = architecture(X)
    # print(f"Got {Y_pred} from architecture")
    loss_val = loss(Y_pred, Y)

    return float(loss_val)
    
def project_onto(weights, directions, flat=False, on_gpu=True):
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

    if not flat:
        flat_weights = flatten(weights)
    else:
        flat_weights = weights

    if on_gpu:
        flat_weights = flat_weights.cpu().numpy()

    for dir in directions:
        dir_pt = {key:torch.from_numpy(dir[key]) for key in dir}
        flat_dir = flatten(dir_pt).cpu().numpy()
        coeff = np.dot(flat_weights, flat_dir) / np.linalg.norm(flat_dir)
        projection_coeffs.append(coeff)

    return projection_coeffs


def cumsum(ordered_dict_list):
    cumsum_list = [ordered_dict_list[0]]
    for elt in ordered_dict_list[1: ]:
        sum_so_far = cumsum_list[-1]
        new_elt = {state_name:sum_so_far[state_name] + elt[state_name]\
            for state_name in elt}
        cumsum_list.append(new_elt)
    
    return cumsum_list