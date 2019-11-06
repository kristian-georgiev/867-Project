#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This example shows how to use higher to do Model Agnostic Meta Learning (MAML)
for few-shot Omniglot classification.
For more details see the original MAML paper:
https://arxiv.org/abs/1703.03400

This code has been modified from Jackie Loong's PyTorch MAML implementation:
https://github.com/dragen1860/MAML-Pytorch/blob/master/omniglot_train.py
"""

import argparse
import time
import typing

import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

import higher

from support.omniglot_loaders import OmniglotNShot
from support.quickdraw_loaders import QuickdrawNShot
import meta_ops
import pdb


def main():
    argparser = argparse.ArgumentParser('Few Shot Learning')
    argparser.add_argument('--dataset', type=str, help='omniglot/miniimagenet/etc.', default='quickdraw')
    argparser.add_argument('--metalearner', type=str, help='maml/reptile/anil/nil/etc.', default='maml')
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--n_epoch', type=int, help='num epochs', default=100)
    argparser.add_argument('--freeze', type=int, help='freeze for anil', default=2)
    argparser.add_argument(
        '--k_spt', type=int, help='k shot for support set', default=5)
    argparser.add_argument(
        '--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument(
        '--task_num',
        type=int,
        help='meta batch size, namely task num',
        default=32)
    argparser.add_argument('--seed', type=int, help='random seed', default=1)
    args = argparser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # Set up the Omniglot loader.
    device = torch.device('cuda')
    if args.dataset == 'omniglot':
        db = OmniglotNShot(
            '/tmp/omniglot-data',
            batchsz=args.task_num,
            n_way=args.n_way,
            k_shot=args.k_spt,
            k_query=args.k_qry,
            imgsz=28,
            device=device,
        )
    elif args.dataset == 'quickdraw': 
        db = QuickdrawNShot(
            './support/data/QuickDrawData.pkl',
            batchsz=args.task_num,
            n_way=args.n_way,
            k_shot=args.k_spt,
            k_query=args.k_qry,
            imgsz=28,
            device=device,
        )

    # Create a vanilla PyTorch neural network that will be
    # automatically monkey-patched by higher later.
    # Before higher, models could *not* be created like this
    # and the parameters needed to be manually updated and copied
    # for the updates.
    if args.dataset in ['omniglot', 'quickdraw']: 
        net = nn.Sequential(
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
            nn.Linear(64, args.n_way)).to(device)

    # We will use Adam to (meta-)optimize the initial parameters
    # to be adapted.
    meta_opt = optim.Adam(net.parameters(), lr=1e-3)

    if args.metalearner == 'maml': 
        train = meta_ops.train_maml
        test  = meta_ops.test_maml
        train_dict = {'db': db, 'net': net, 'device': device, 'meta_opt': meta_opt}
        test_dict = {'db': db, 'net': net, 'device': device}
    elif args.metalearner == 'anil': 
        train = meta_ops.train_anil
        test  = meta_ops.test_maml
        train_dict = {'db': db, 'net': net, 'device': device, 'meta_opt': meta_opt, 'freeze': args.freeze}
        test_dict = {'db': db, 'net': net, 'device': device}

    log = []
    weights_across_training = [net.state_dict()]
    for epoch in range(args.n_epoch):
        train_dict['epoch'] = epoch 
        train_dict['log'] = log
        test_dict['epoch'] = epoch
        test_dict['log'] = log
        train(**train_dict)
        test(**test_dict)
        plot(log, args)

        previous_weights = weights_across_training[-1]
        new_weights = net.state_dict()
        gradient_update = {key: new_weights.get(key, 0) - previous_weights[key] for key in previous_weights.keys()}
        weights_across_training.append(gradient_update)

    np.save("./" + args.metalearner + "/" + args.dataset + "/gradient_updates.npy", np.array(weights_across_training))
    torch.save(net.state_dict(), "./"  + args.metalearner + "/" + args.dataset + "/model_state_dict.pt")


def plot(log, args):
    # Generally you should pull your plotting code out of your training
    # script but we are doing it here for brevity.
    df = pd.DataFrame(log)

    fig, ax = plt.subplots(figsize=(6, 4))
    train_df = df[df['mode'] == 'train']
    test_df = df[df['mode'] == 'test']
    ax.plot(train_df['epoch'], train_df['acc'], label='Train')
    ax.plot(test_df['epoch'], test_df['acc'], label='Test')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(60, 100)
    fig.legend(ncol=2, loc='lower right')
    fig.tight_layout()
    plt.title(args.metalearner + ' on ' + args.dataset)
    fname = args.metalearner + '_' + args.dataset + '.png'
    print(f'--- Plotting accuracy to {fname}')
    fig.savefig(fname)
    plt.close(fig)


# Won't need this after this PR is merged in:
# https://github.com/pytorch/pytorch/pull/22245
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


if __name__ == '__main__':
    main()
