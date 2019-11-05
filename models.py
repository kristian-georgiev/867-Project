import torch
from torch import nn
import torch.nn.functional as F

def model(modelname):
    models = {"small" : small,
            "medium": medium,
            "large" : large}

    return models[modelname]

def small():
    pass

def medium():
    pass

def large():
    pass

