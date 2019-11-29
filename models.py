import torch
from torch import nn
import torch.nn.functional as F

def modelloader(modelname):
    modelloaders = {"mini" : mini,
                    "small_no_batch_norm" : small_no_batch_norm,
                    "small" : small,
                    "medium": medium,
                    "large" : large}

    return modelloaders[modelname]

# for debugging purposes
def mini(hparams):
        return nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=10),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            Flatten(),
            nn.Linear(2, hparams.n_way)).to(hparams.device)



def small_no_batch_norm(hparams):
        return nn.Sequential(
            nn.Conv2d(1, 64, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 64, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 64, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            Flatten(),
            nn.Linear(64, hparams.n_way)).to(hparams.device)


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
