#!/usr/bin/env python3

"""
Main file for our 6.867 project on meta-learning

Instructions on how to run this file and modify any
hyperparameters are in README.md
"""

# This file is a modification of the example given in the 
# repository for the higher library. The original can be found at
# https://github.com/facebookresearch/higher/blob/master/examples/maml-omniglot.py

# below is the description of the file we adapted
"""
Copyright (c) Facebook, Inc. and its affiliates.
Licensed under the Apache License, Version 2.0 (the "License");

This example shows how to use higher to do Model Agnostic Meta Learning (MAML)
for few-shot Omniglot classification.
For more details see the original MAML paper:
https://arxiv.org/abs/1703.03400

This code has been modified from Jackie Loong's PyTorch MAML implementation:
https://github.com/dragen1860/MAML-Pytorch/blob/master/omniglot_train.py
"""

import sys
import time

if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")

# for runtime type-checking
import typing

# for class-like beaviour of parameter dictionaries
# i.e. hparams.batch_size, not hparams["batch_size"]
from munch import Munch

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

import higher
import copy

# local imports
from parse import parse
import models
import meta_learners
import plotter
import plotting_util
import dataloader as dl
import pdb
import argparse


argparser = argparse.ArgumentParser()
argparser.add_argument('--model_training', 
                       choices=["train_new", "pretrained"], required=True,
                       help="Choose whether to train a new model or load a \
                             pretrained one.")
argparser.add_argument('--meta_learner',
                       choices=["maml", "anil", "sgd"],
                       help="Choose meta-learning algorithm to follow.\
                             If sgd is chosen, simply joint training is performed.")
argparser.add_argument('--dataset', choices=["omniglot", "quickdraw"])
argparser.add_argument('--hparams_file', type=str, default="./hparams.yaml")
argparser.add_argument('--loss_plots_dir', type=str, default="./plots")
argparser.add_argument('--use_gpu', type=bool, default=True)

args = argparser.parse_args()

# parse hyperparameters
hparams = Munch(parse(args.hparams_file)[args.model_training])
if args.meta_learner:
    hparams.meta_learner = args.meta_learner


if args.dataset:
    hparams.dataset = args.dataset

hparams.modelpath += "/" + hparams.meta_learner + "/" + hparams.dataset + "/" +\
    hparams.model + "/" + hparams.modelname

hparams.trajpath += "/" + hparams.meta_learner + "/" + hparams.dataset + "/" +\
    hparams.model + "/" + hparams.weightstrajfilename


print(hparams)

# set random seed
torch.manual_seed(hparams.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(hparams.seed)
np.random.seed(hparams.seed)

# init CPU / GPU
device = torch.device("cuda:0" if torch.cuda.is_available() and args.use_gpu else "cpu")
hparams.update(device=device)
print(f"Device is {hparams.device}.")

if not args.model_training == "pretrained":
    # init model
    modelloader = models.modelloader(hparams.model)
    net = modelloader(hparams)

    # init optimizer
    if hparams.optimizer == "adam":
        meta_optim = torch.optim.Adam(net.parameters(), lr=hparams.learning_rate)

    # init dataloader
    dataloader = dl.dataloader(hparams)

    # init meta-learning algorithm
    metalearner = meta_learners.meta_learner(hparams)
    print(f"Using {metalearner.toString()} as a meta-learning algorithm.")
    train, test = metalearner.get_train_and_test()

    train_dict = {'db': dataloader, 
                'net': net, 
                'device': device, 
                'meta_opt': meta_optim,
                'lr_finetune': hparams.lr_finetune}
    test_dict = {'db': dataloader, 
                'net': net, 
                'device': device,
                'lr_finetune': hparams.lr_finetune}


    # -----------------
    # train & test loop
    # -----------------
    log = []
    weights_over_time = [copy.deepcopy(net.state_dict())]

    print(f"Training for {hparams.num_epochs} epochs.")
    for epoch in range(hparams.num_epochs):
        train_dict['epoch'] = epoch 
        train_dict['log'] = log
        test_dict['epoch'] = epoch
        test_dict['log'] = log
        
        train(**train_dict)
        test(**test_dict)

        if hparams.saving_gradient_steps:
            weights_over_time.append(copy.deepcopy(net.state_dict()))

        if hparams.plot_progress:
            plotter.plot_progress(log)

    # serialize model
    np.save(hparams.trajpath, np.array(weights_over_time))
    torch.save(net.state_dict(), hparams.modelpath)

    # load model state dictionary
    state_dict = copy.deepcopy(net.state_dict())

    for key in state_dict:
        assert torch.all(torch.eq(state_dict[key], weights_over_time[-1][key]))

else: # args.model_training == "pretrained"

    # load model state dictionary
    state_dict = torch.load(hparams.modelpath)

    # init model
    modelloader = models.modelloader(hparams.model)
    net = modelloader(hparams)
    net.load_state_dict(state_dict)

    # load gradient updates
    weights_over_time = np.load(hparams.trajpath, allow_pickle=True)
    print("Loaded model and gradient updates!")

if hparams.loss_plotting:


    for n in state_dict:
        print(f"Layer {n}, shape {state_dict[n].shape}")

    # sanity check
    # ==================================================
    net2 = models.modelloader(hparams.model)(hparams)
    net2.load_state_dict(copy.deepcopy(state_dict))
    for key in net.state_dict():
        torch.eq(net.state_dict()[key], net2.state_dict()[key])
    # ==================================================

    # init dataloader
    dataloader = dl.dataloader(hparams)


    for i in range(hparams.index):
        test_dataset = dataloader.next(mode='test')
    _, __, X, Y = test_dataset # take only query dataset
    test_dataset = (X, Y)

    plotting_util.plot_images(X[0])
    # pdb.set_trace()

    # define loss
    loss = F.cross_entropy

    # sanity check
    # make sure end loss is small 
    # ==================================================
    net.eval()    
    print([i for i in net.parameters()][0][0][0])
    with torch.no_grad(): 
        Y_pred = net(X[0])
    print("loss from sanity check is:", loss(Y_pred, Y[0]))
    # ==================================================


    Ws = plotting_util.state_dicts_list_to_numpy_array(weights_over_time)
    index_weights_to_take = len(Ws) - hparams.last_n_traj_points - 1
    Ws = Ws[index_weights_to_take:]

    weight_shapes = plotting_util.get_shapes_indices(weights_over_time[0])
    state_dict_template = weights_over_time[0]


    # sanity check 
    # ==================================================
    # conversion from dict of tensors to numpy array and back
    to_np_and_back = plotting_util.numpy_array_to_state_dict(plotting_util.state_dicts_list_to_numpy_array([state_dict])[0], weight_shapes, state_dict_template)
    for n in state_dict:
        assert torch.all(torch.eq(state_dict[n].cpu(), to_np_and_back[n]))
    # ==================================================


    # sanity check 
    # loss over trajectory
    # ==================================================
    print("=========================")
    print(f"weights over time are {Ws[:,0:5]}")
    print("=========================")

    for i in range (len(list(Ws))):
        w = Ws[i]
        sd = plotting_util.numpy_array_to_state_dict(w, weight_shapes, state_dict_template)
        net.load_state_dict(sd)
        net.eval()

        with torch.no_grad():
            Y_pred = net(X[0])
            print(f"loss from trajectory elt {i} is:", loss(Y_pred, Y[0]))
    # ==================================================



    # sanity check
    # random point
    # ==================================================
    sh = Ws[0].shape
    r = np.random.random_sample(sh)
    sd = plotting_util.numpy_array_to_state_dict(r, weight_shapes, state_dict_template)
    net.load_state_dict(sd)
    net.eval()
    with torch.no_grad():
        Y_pred = net(X[0])
        print(f"loss from a random point is {loss(Y_pred, Y[0])}")
    # ==================================================


    # sanity check
    # ==================================================
    # make sure weights from cumulative sum agree with final weights
    final_weights = plotting_util.state_dicts_list_to_numpy_array([state_dict])
    assert np.allclose(final_weights[-1], Ws[-1])
    # ==================================================

    # get directions from weights over time
    directions = plotter.pca_directions(Ws)
    print(directions[:, 0:5])
    print(f"Got PCA directions!")

    ml = models.modelloader(hparams.model)

    plot_filename = plotter.plot_loss_landscape(directions,
                                                test_dataset,
                                                ml,
                                                loss,
                                                Ws,
                                                weight_shapes,
                                                state_dict_template,
                                                args.loss_plots_dir,
                                                hparams)
    print(f"Saved plots in {args.loss_plots_dir}/{plot_filename}")
