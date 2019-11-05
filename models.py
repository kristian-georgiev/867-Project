import torch
from torch import nn
import torch.nn.functional as F

def modelloader(modelname):
    modelloaders = {"small" : small,
                    "medium": medium,
                    "large" : large}

    return modelloaders[modelname]

def small(hparams):
        return nn.Sequential(
            nn.Conv2d(1, 64, 3),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            Flatten(),
            nn.Linear(64, hparams.n_way)).to(hparams.device)

def medium():
    pass

def large():
    pass

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
