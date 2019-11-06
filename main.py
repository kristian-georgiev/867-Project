#!/usr/bin/env python3

"""
Main file for our 6.867 project on meta-learning

Instructions on how to run this file and modify any
hyperparameters are in README.md
"""

# This file is a modification of the example given in the 
# repository for the higher library. The original can be found at
# https://github.com/facebookresearch/higher/blob/master/examples/maml-omniglot.py

# below is the description of the 
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

# local imports
from parse import parse
import models
import meta_learners
import plotter
import dataloader

# parse config
config_choice = sys.argv[1].rstrip()
config = parse("config.yaml")[config_choice]
config = Munch(config)
hparams_choice = config.parameters_choice

# parse hyperparameters
hparams = Munch(parse(config.parameters_file)[hparams_choice])

# set random seed
torch.manual_seed(hparams.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(hparams.seed)
np.random.seed(hparams.seed)

# init CPU / GPU
device = torch.device("cuda:0" if torch.cuda.is_available() and config.gpu_available else "cpu")
hparams.update(device=device)
print(f"Device is {hparams.device}.")

if not config.parameters_choice == "pretrained":
    # init model
    modelloader = models.modelloader(hparams.model)
    net = modelloader(hparams)

    # init optimizer
    if hparams.optimizer == "adam":
        meta_optim = torch.optim.Adam(net.parameters(), lr=hparams.learning_rate)

    # init dataloader
    dataloader = dataloader.dataloader(hparams)

    # init meta-learning algorithm
    metalearner = meta_learners.meta_learner(hparams)
    print(f"Using {metalearner.toString()} as a meta-learning algorithm.")
    train, test = metalearner.get_train_and_test()

    train_dict = {'db': dataloader, 
                'net': net, 
                'device': device, 
                'meta_opt': meta_optim}
    test_dict = {'db': dataloader, 
                'net': net, 
                'device': device}


    # -----------------
    # train & test loop
    # -----------------
    log = []
    weights_accross_training = [net.state_dict()]

    for epoch in range(hparams.num_epochs):
        train_dict['epoch'] = epoch 
        train_dict['log'] = log
        test_dict['epoch'] = epoch
        test_dict['log'] = log
        
        train(**train_dict)
        test(**test_dict)

        if hparams.saving_gradient_steps:
            previous_weights = weights_accross_training[-1]
            new_weights = net.state_dict()
            gradient_update = {key: new_weights.get(key, 0) - previous_weights[key] for key in previous_weights.keys()}
            weights_accross_training.append(gradient_update)

        if hparams.plot_progress:
            plotter.plot_progress(log)

    print(net)

    # serialize model
    np.save("./models/gradient_updates.npy", np.array(weights_accross_training))
    torch.save(net.state_dict(), hparams.modelpath)

else: # config.parameters_choice == "pretrained"

    # load model state dictionary
    state_dict = torch.load(hparams.modelpath)

    # init model
    modelloader = models.modelloader(hparams.model)
    net = modelloader(hparams)
    net.load_state_dict(state_dict)

    # load gradient updates
    updates_accross_training = np.load("./models/gradient_updates.npy", allow_pickle=True)
    print("Loaded model and gradient_updates!")


if hparams.loss_plotting:

    # remove non-weight/bias elements from state dictionaries
    # e.g. running averages go there as well, etc.
    for i in range(len(updates_accross_training)):
        updates_accross_training[i] = {state_name:updates_accross_training[i][state_name]\
        for state_name in updates_accross_training[i] \
        if "weight" in state_name or "bias" in state_name}

    directions = plotter.pca_directions(updates_accross_training)
    print(f"Got PCA directions!")

    # init dataloader
    dataloader = dataloader.dataloader(hparams)

    test_dataset = dataloader.next()
    print(len(test_dataset))

    # define loss
    loss = F.cross_entropy

    plot_filename = plotter.plot_loss_landscape(directions, test_dataset, net, loss, hparams.plot_gridsize, weights_over_time)
    print(f"Saved plots in {config.loss_plots_dir}/{plot_filename}")
