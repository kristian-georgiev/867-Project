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
import copy

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
    # shapes looks like [num_params_in_filter_1, num_params_in_filter_2, ...]
    ind = np.cumsum(shapes)
    result = [(0, ind[0])]
    for i in range(len(ind) - 1):
        result.append((ind[i], ind[i + 1]))

    # result, a list of tuples, looks like 
    # [(0:num_params_in_filter_1), (num_params_in_filter_1: num_params_in_filter_2), ...]
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

def loss_eval(i, j, offset, 
              loss, directions,
              X, Y,
              ml,
              shapes,
              state_dict_template,
              hparams):
    assert len(directions) == 2
    weights = i * directions[0] + j * directions[1] + offset

    # go from flat np array to an ordered dict state_dict with
    # all the net structure 
    new_state = numpy_array_to_state_dict(weights,
                                          shapes, 
                                          state_dict_template)


    net = ml(hparams)
    net.load_state_dict(new_state)
    opt = torch.optim.Adam(net.parameters(), lr=hparams.lr_finetune)

    net.eval()
    with torch.no_grad(): 
        Y_pred = net(X)
    init_loss = loss(Y_pred, Y)

    net.train()
    for i in range(hparams.n_inner_iter):
        predictions = net(X)
        ft_loss = F.cross_entropy(predictions, Y)
        ft_loss.backward()
        opt.step()

    net.eval()

    finetuned_state = copy.deepcopy(net.state_dict())
    finetuned_weights = state_dicts_list_to_numpy_array([finetuned_state])[0]
    update_magnitude = np.sum(get_rescaling_factors(finetuned_weights - weights, shapes)) # sum of Frob. norms of filters/layers
    projected_vector_update = project_onto(finetuned_weights, directions, offset)

    with torch.no_grad(): 
        Y_pred = net(X)
    finetuned_loss = loss(Y_pred, Y)    

    return float(init_loss), float(finetuned_loss), update_magnitude, tuple(projected_vector_update)


def plot_images(batch, labels, dataset):
    print("inside plot images")
    for k, im in enumerate(batch): 
        pixels = np.reshape(im.cpu().numpy(), (28, 28))
        plt.imshow(pixels, cmap='gray')
        label = str(labels[k].cpu().numpy())
        plt.title(label)
        plt.savefig('plots/data/' + dataset + '/' + label + '_' + str(k) + '.png')
        plt.close()
