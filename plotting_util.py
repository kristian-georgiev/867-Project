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

    # remove non-weight/bias elements from state dictionaries
    # e.g. running averages go there as well, etc.
    # print("before", type(state_dicts), state_dicts.shape, len(state_dicts[0]))

    # print(state_dicts[0].keys())

    # for i in range(len(state_dicts)):
    #     state_dicts[i] = {state_name:state_dicts[i][state_name]\
    #     for state_name in state_dicts[i] \
    #     if "weight" in state_name or "bias" in state_name}

    # print("after", type(state_dicts), state_dicts.shape, len(state_dicts[0]))
    result = np.vstack([flatten(d, torch=False) for d in state_dicts])
    # print(result.shape)
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



# def unflatten(dirs, weights_with_desired_shape):
#     assert (len(flatten(weights_with_desired_shape)) == len(dirs[0])),\
#          f"We need dimensions {len(dirs)} of dirs and \
#              {len(flatten(weights_with_desired_shape))} of state_dict to match up"

#     unflattened_dirs = []

#     for d in dirs:
#         unflattened_dir = OrderedDict()
        
#         for w in weights_with_desired_shape:
#             l = len(weights_with_desired_shape[w].reshape(-1))
#             shape = weights_with_desired_shape[w].shape
#             unflattened_dir[w] = d[ :l].reshape(shape)
#             d = d[l: ]
#         unflattened_dirs.append(unflattened_dir)
        
#     return unflattened_dirs

def get_shapes_indices(weights_dict):
    shapes = [np.prod(weights_dict[t].shape) for t in weights_dict]
    # shapes looks like [num_params_in_layer_1, num_params_in_layer_2, ...]
    ind = np.cumsum(shapes)
    result = [(0, ind[0])]
    for i in range(len(ind) - 1):
        result.append((ind[i], ind[i + 1]))

    # result looks like 
    # [(0:num_params_in_layer_1), (num_params_in_layer_1: num_params_in_layer_2), ...]
    return result

def flatten(weights_dict, torch=False):
    flat_weights = [weights_dict[t].reshape(-1) for t in weights_dict]
    if torch:
        return torch.cat(flat_weights)
    else:
        flat_weights = [x.cpu().numpy() for x in flat_weights]
        return np.concatenate(flat_weights)

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
    # weights = {key: theta_star[key].cpu().numpy() + i * directions[0][key] + j * directions[1][key] \
    #     for key in directions[0].keys()}
    # weights = {key: theta_star[key].cpu().numpy() \
    #     for key in directions[0].keys()}
    # weights = theta_star

    # old_state = architecture.state_dict()

    assert len(directions) == 2
    weights = i * directions[0] + j * directions[1] + offset

    # print("~~~~~~~~~~~")
    # print(offset[0:5])
    # print(weights[0:5])
    # print("~~~~~~~~~~~")

    if abs(i) < 0.001 and abs(j) < 0.001:
        print(i, j, np.allclose(weights, offset))

    # go from flat np array to an ordered dict state_dict with
    # all the net structure 
    new_state = numpy_array_to_state_dict(weights,
                                          shapes, 
                                          state_dict_template)

    # # update keys that are present in weights
    # # for context, net.state_dict() contains tensors that are
    # # neither weights, nor biases, and they are not present in the 
    # # weights ordered dictionary
    # for key, old_val in old_state.items():
    #     if key in weights.keys():
    #         if isinstance(weights[key], torch.Tensor):
    #             new_state[key] = weights[key]
    #         else:
    #             new_state[key] = torch.from_numpy(weights[key])
    #     else:
    #         new_state[key] = old_val

    # print("before loading", [i for i in architecture.parameters()][0][0][0])


    architecture.load_state_dict(new_state)
    architecture.eval()
    
    # print("after loading", [i for i in architecture.parameters()][0][0][0])
    # print("-------------------")

    with torch.no_grad(): 
        Y_pred = architecture(X)
    # print(f"Got {Y_pred} from architecture")
    loss_val = loss(Y_pred, Y)

    return float(loss_val)
    
def project_onto(weights, directions, offset):
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
    assert len(weights) == len(directions[0])

    projection_coeffs = []

    for d in directions:
        v = weights - offset
        coeff = np.dot(v, d) / np.linalg.norm(d)
        projection_coeffs.append(coeff)

    return projection_coeffs


# def cumsum(ordered_dict_list):
#     cumsum_list = [ordered_dict_list[0]]
#     for elt in ordered_dict_list[1: ]:
#         sum_so_far = cumsum_list[-1]
#         new_elt = {state_name:sum_so_far[state_name] + elt[state_name] \
#             for state_name in elt}
#         cumsum_list.append(new_elt)
#     # for i in range(len(cumsum_list)): 
#     #     for key in cumsum_list[i].keys(): 
#     #         cumsum_list[i][key] -= cumsum_list[-1][key]

#     # return cumsum_list[:-1]
#     return cumsum_list[:-1]