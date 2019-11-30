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

def state_dicts_list_to_numpy_array(state_dicts):
    def flatten(weights_dict):
        flat_weights = [weights_dict[t].reshape(-1) for t in weights_dict]
        flat_weights = [x.cpu().numpy() for x in flat_weights]
        return np.concatenate(flat_weights)

    result = np.vstack([flatten(d) for d in state_dicts])
    return result

def numpy_array_to_state_dict(arr, shapes, state_dict_template):
    assert len(shapes) == len(state_dict_template)
    n = len(shapes)
    i = 0
    keys = {}
    for key in state_dict_template:
        keys[i] = key
        i += 1

    result = OrderedDict()
    for i in range(n):
        l, r = shapes[i]
        layer_weights = arr[l:r]
        shape = state_dict_template[keys[i]].shape
        layer_weights_np = np.array(layer_weights).reshape(shape)
        result[keys[i]] = torch.from_numpy(layer_weights_np)
        
    return result

def get_shapes_indices(weights_dict):
    shapes = [np.prod(weights_dict[t].shape) for t in weights_dict]
    # shapes looks like [num_params_in_layer_1, num_params_in_layer_2, ...]
    ind = np.cumsum(shapes)
    result = [(0, ind[0])]
    for i in range(len(ind) - 1):
        result.append((ind[i], ind[i + 1]))

    # result, a list of tuples, looks like 
    # [(0:num_params_in_layer_1), (num_params_in_layer_1: num_params_in_layer_2), ...]
    return result

def get_rescaling_factors(arr, shapes):
    # gets Frobenious norms of flattened matrices
    return [np.linalg.norm(arr[i:j]) for (i, j) in shapes] 

def multiply_filterwise(arr, shapes, multipliers):
    assert len(multipliers) == len(shapes)

    num_repeats = [j - i for (i, j) in shapes]
    print(np.array(num_repeats))
    m = np.repeat(multipliers, num_repeats)

    assert len(m) == len(arr)

    return arr * m

def project_onto(weights, directions, offset):
    assert len(weights) == len(directions[0])

    projection_coeffs = []

    for d in directions:
        v = weights - offset
        coeff = np.dot(v, d) / np.linalg.norm(d)
        projection_coeffs.append(coeff)

    return projection_coeffs

def loss_eval(i, j, offset, 
              loss, directions, 
              X, Y, 
              architecture,
              shapes,
              state_dict_template):
    """Evaluate loss on test set at the point given by i, j, directions
    
    Arguments:
        i {float} -- coeff. for first direction, in [-1, 1]
        j {float} -- coeff. for second direction, in [-1, 1]
        offset -- offset of the affine subspace spanned by the directions
        loss {function} -- loss f-n to be evaluated
        directions {list(OrderedDict)} -- list of two directions we got from PCA
        test_dataset {tuple} -- tuple of (X, Y) loaded from dataloader
        architecture {torch.Sequential} -- PyTorch net
    
    Returns:
        float -- loss evaluated at 
        architecture(weights = i * dir[0] + j * dir[1]) on test_dataset
    """
    assert len(directions) == 2
    weights = i * directions[0] + j * directions[1] + offset

    # go from flat np array to an ordered dict state_dict with
    # all the net structure 
    new_state = numpy_array_to_state_dict(weights,
                                          shapes, 
                                          state_dict_template)

    architecture.load_state_dict(new_state)
    architecture.eval()
    
    with torch.no_grad(): 
        Y_pred = architecture(X)
    loss_val = loss(Y_pred, Y)

    return float(loss_val)