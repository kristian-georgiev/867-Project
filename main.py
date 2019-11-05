#!/usr/bin/env python3

# Main file for our 6.867 project on meta-learning

# Instructions on how to run this file and modify any
# hyperparameters are in README.md

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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')

import torch
from torch import nn
import torch.nn.functional as F

import higher

# local imports
from parse import parse
from support.omniglot_loaders import OmniglotNShot
import meta_ops


# parse config
config_choice = sys.argv[1].rstrip()
config = parse("config.yaml")[config_choice]
config = Munch(config)
hparams_choice = config.parameters_choice

# parse hyperparameters
hparams = parse(config.parameters_file)[hparams_choice]

device = torch.device("cuda:0" if torch.cuda.is_available() and config.gpu_available else "cpu")
