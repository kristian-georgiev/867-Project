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

def state_dicts_list_to_numpy_array(state_dicts, fix_extractor, fix_head):
    def flatten(weights_dict):
        flat_weights = [weights_dict[t].reshape(-1) for t in weights_dict  if ("weight" in t or "bias" in t)]
        if fix_extractor: 
            # fixes the extractor, hence it only varies the head
            flat_weights = flat_weights[-2:]
        elif fix_head: 
            flat_weights = flat_weights[:-2]
        # print([t for t in weights_dict if ("weight" in t or "bias" in t)])
        flat_weights = [x.cpu().numpy() for x in flat_weights]
        return np.concatenate(flat_weights)

    result = np.vstack([flatten(d) for d in state_dicts]) # ignore batch norm params / statistics
    return result

def numpy_array_to_state_dict(arr, shapes, state_dict_template, hparams):
    assert len(shapes) <= len(state_dict_template)
    n = len(shapes)
    i = 0
    keys = {}
    result = OrderedDict()

    length_dict = len(state_dict_template)
    for ind, key in enumerate(state_dict_template):
        condition = ("weight" in key or "bias" in key)
        if hparams.fix_extractor:
            condition = (condition and ind >= length_dict - 2)
        elif hparams.fix_head: 
            condition = (condition and ind < length_dict - 2)
        if condition:
            keys[i] = key
            i += 1
        else:
            result[key] = state_dict_template[key] # copy BN statistics from template (final weights)

    for i in range(n):
        l, r = shapes[i]
        layer_weights = arr[l:r]
        shape = state_dict_template[keys[i]].shape
        layer_weights_np = np.array(layer_weights).reshape(shape)
        result[keys[i]] = torch.from_numpy(layer_weights_np)
        
    return result

def get_shapes_indices(weights_dict, fix_extractor, fix_head):
    shapes = [np.prod(weights_dict[t].shape) for t in weights_dict if ("weight" in t or "bias" in t)]
    # shapes looks like [num_params_in_filter_1, num_params_in_filter_2, ...]
    if fix_extractor:
        # fixes the extractor, hence it only varies the head
        shapes = shapes[-2:]
    elif fix_head: 
        shapes = shapes[:-2]
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
              X_s_b, Y_s_b,
              X_b, Y_b,
              ml,
              shapes,
              state_dict_template,
              hparams):
    assert len(directions) == 2
    querysz = X_b[0].size(0)
    weights = i * directions[0] + j * directions[1] + offset

    # go from flat np array to an ordered dict state_dict with
    # all the net structure 
    new_state = numpy_array_to_state_dict(weights,
                                          shapes, 
                                          state_dict_template,
                                          hparams)


    init_losses = []
    finetuned_losses = []
    finetuned_weights_list = []
    update_magnitudes = []
    projected_vector_updates = []
    accuracies = []

    for X, Y, X_s, Y_s in zip(X_b, Y_b, X_s_b, Y_s_b):
        net = ml(hparams)
        net.load_state_dict(new_state)
        opt = torch.optim.SGD(net.parameters(), lr=hparams.lr_finetune)

        net.eval()
        with torch.no_grad(): 
            Y_pred = net(X)
        init_losses.append(loss(Y_pred, Y))

        net.train()
        for i in range(hparams.n_inner_iter):
            predictions = net(X_s)
            ft_loss = F.cross_entropy(predictions, Y_s)
            ft_loss.backward()
            opt.step()

        net.eval()

        finetuned_state = copy.deepcopy(net.state_dict())
        finetuned_weights = state_dicts_list_to_numpy_array([finetuned_state], hparams.fix_extractor, hparams.fix_head)[0]
        finetuned_weights_list.append(finetuned_weights)
        update_magnitudes.append(np.sum(get_rescaling_factors(finetuned_weights - weights, shapes))) # sum of Frob. norms of filters/layers
        projected_vector_updates.append(list(np.dot(finetuned_weights.reshape(1, -1) - weights.reshape(1, -1), directions.T)[0]))

        with torch.no_grad(): 
            Y_pred = net(X)
        finetuned_losses.append(loss(Y_pred, Y))
        accuracies.append((Y_pred.argmax(dim=1) == Y).sum().item() / querysz)

    init_loss = np.mean([float(elt) for elt in init_losses])
    finetuned_loss = np.mean([float(elt) for elt in finetuned_losses])
    update_magnitude = np.mean(update_magnitudes)
    projected_vector_update = np.mean(projected_vector_updates, axis=0)
    accuracy = np.mean(accuracies)

    return float(init_loss), float(finetuned_loss), update_magnitude, tuple(projected_vector_update), float(accuracy)


def plot_images(batch, labels, dataset):
    for k, im in enumerate(batch): 
        pixels = np.reshape(im.cpu().numpy(), (28, 28))
        plt.imshow(pixels, cmap='gray')
        label = str(labels[k].cpu().numpy())
        plt.title(label)
        plt.savefig('plots/data/' + dataset + '/' + label + '_' + str(k) + '.png')
        plt.close()
