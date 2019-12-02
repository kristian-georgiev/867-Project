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

# import pandas as pd
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

import pdb

def train_sgd(db, net, device, meta_opt, epoch, log):
    # call this before starting to train: this enables modules like dropout.
    net.train()
    n_train_iter = db.x_train.shape[0] // db.batchsz

    for batch_idx in range(n_train_iter):
        start_time = time.time()
        # Sample a batch of support and query images and labels.
        x_spt, y_spt, x_qry, y_qry = db.next()

        task_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)

        # Initialize the inner optimizer to adapt the parameters to
        # the support set.
        optimizer = torch.optim.SGD(net.parameters(), lr=1e-1)
        # thus the inner optimizer will do 5 steps of SGD to get the fast weights

        qry_losses = []
        qry_accs = []
        optimizer.zero_grad()

        # so task_num is the meta batch
        for i in range(task_num):
            # now we are at task i: we have 32 tasks with 5 sampled classes
            # (5-way) and 5 examples from each class (5-shot)

            spt_logits = net(x_spt[i])
            spt_loss = F.cross_entropy(spt_logits, y_spt[i])

            # the query has N_way x K_query examples, in this case 5 x 15 = 75
            qry_logits = net(x_qry[i])
            qry_loss = F.cross_entropy(qry_logits, y_qry[i])
            qry_losses.append(qry_loss.detach())
            qry_acc = (qry_logits.argmax(
                dim=1) == y_qry[i]).sum().item() / querysz
            qry_accs.append(qry_acc)

            spt_loss.backward()

        optimizer.step()

        qry_losses = sum(qry_losses) / task_num
        qry_accs = 100. * sum(qry_accs) / task_num
        i = epoch + float(batch_idx) / n_train_iter
        iter_time = time.time() - start_time
        if batch_idx % 4 == 0:
            print(f'[Epoch {i:.5f}] Train Loss: {qry_losses:.5f} | Acc: {qry_accs:.5f} | Time: {iter_time:.5f}')

        log.append({
            'epoch': i,
            'loss': qry_losses,
            'acc': qry_accs,
            'mode': 'train',
            'time': time.time(),
        })

def train_maml(db, net, device, meta_opt, lr_finetune, epoch, log):
    # call this before starting to train: this enables modules like dropout.
    net.train()
    n_train_iter = db.x_train.shape[0] // db.batchsz

    for batch_idx in range(n_train_iter):
        start_time = time.time()
        # Sample a batch of support and query images and labels.
        x_spt, y_spt, x_qry, y_qry = db.next()

        task_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)

        # TODO: Maybe pull this out into a separate module so it
        # doesn't have to be duplicated between `train` and `test`?

        # Initialize the inner optimizer to adapt the parameters to
        # the support set.
        n_inner_iter = 5
        inner_opt = torch.optim.Adam(net.parameters(), lr=lr_finetune)
        # thus the inner optimizer will do 5 steps of SGD to get the fast weights

        qry_losses = []
        qry_accs = []
        meta_opt.zero_grad()
        # so task_num is the meta batch
        for i in range(task_num):
            # now we are at task i: we have 32 tasks with 5 sampled classes 
            # (5-way) and 5 examples from each class (5-shot)
            with higher.innerloop_ctx(net, inner_opt, copy_initial_weights=False) as (fnet, diffopt):
                # Optimize the likelihood of the support set by taking
                # gradient steps w.r.t. the model's parameters.
                # This adapts the model's meta-parameters to the task.
                # higher is able to automatically keep copies of
                # your network's parameters as they are being updated.
                for _ in range(n_inner_iter):
                    spt_logits = fnet(x_spt[i])
                    spt_loss = F.cross_entropy(spt_logits, y_spt[i])
                    diffopt.step(spt_loss)

                # The final set of adapted parameters will induce some
                # final loss and accuracy on the query dataset.
                # These will be used to update the model's meta-parameters.
                
                # the query has N_way x K_query examples, in this case 5 x 15 = 75
                qry_logits = fnet(x_qry[i])
                qry_loss = F.cross_entropy(qry_logits, y_qry[i])
                qry_losses.append(qry_loss.detach())
                qry_acc = (qry_logits.argmax(
                    dim=1) == y_qry[i]).sum().item() / querysz
                qry_accs.append(qry_acc)

                # Update the model's meta-parameters to optimize the query
                # losses across all of the tasks sampled in this batch.
                # This unrolls through the gradient steps.
                qry_loss.backward()

        meta_opt.step()

        qry_losses = sum(qry_losses) / task_num
        qry_accs = 100. * sum(qry_accs) / task_num
        i = epoch + float(batch_idx) / n_train_iter
        iter_time = time.time() - start_time
        if batch_idx % 4 == 0:
            print(f'[Epoch {i:.2f}] Train Loss: {qry_losses:.2f} | Acc: {qry_accs:.2f} | Time: {iter_time:.2f}')

        log.append({
            'epoch': i,
            'loss': qry_losses,
            'acc': qry_accs,
            'mode': 'train',
            'time': time.time(),
        })

def train_anil(db, net, device, meta_opt, epoch, log, freeze):
    """
        freeze: up until which layer we should freeze
    """

    # call this before starting to train: this enables modules like dropout.
    net.train()
    n_train_iter = db.x_train.shape[0] // db.batchsz

    for batch_idx in range(n_train_iter):
        start_time = time.time()
        # Sample a batch of support and query images and labels.
        x_spt, y_spt, x_qry, y_qry = db.next()

        task_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)

        # TODO: Maybe pull this out into a separate module so it
        # doesn't have to be duplicated between `train` and `test`?

        # Set up the parameters that will be updated with ANIL
        anil_parameters = []
        cnt = 0
        for m in net.modules():
            if m.__class__.__name__ == 'Sequential': 
                continue
            if m.__class__.__name__ == 'Conv2d':
                if cnt < freeze: 
                    cnt += 1
                    continue
            anil_parameters.append({'params': m.parameters()})

        # Initialize the inner optimizer to adapt the parameters to
        # the support set.
        n_inner_iter = 5
        inner_opt = torch.optim.SGD(anil_parameters, lr=1e-1)

        # thus the inner optimizer will do 5 steps of SGD to get the fast weights

        qry_losses = []
        qry_accs = []
        meta_opt.zero_grad()
        # so task_num is the meta batch
        for i in range(task_num):
            # so now we are at task i: we have 32 tasks with 5 sampled classes 
            # (5-way) and 5 examples from each class (5-shot)
            with higher.innerloop_ctx(
                net, inner_opt, copy_initial_weights=False
            ) as (fnet, diffopt):
                # Optimize the likelihood of the support set by taking
                # gradient steps w.r.t. the model's parameters.
                # This adapts the model's meta-parameters to the task.
                # higher is able to automatically keep copies of
                # your network's parameters as they are being updated.
                for _ in range(n_inner_iter):
                    spt_logits = fnet(x_spt[i])
                    spt_loss = F.cross_entropy(spt_logits, y_spt[i])
                    diffopt.step(spt_loss)

                # The final set of adapted parameters will induce some
                # final loss and accuracy on the query dataset.
                # These will be used to update the model's meta-parameters.
                
                # the query has N_way x K_query examples, in this case 5 x 15 = 75
                qry_logits = fnet(x_qry[i])
                qry_loss = F.cross_entropy(qry_logits, y_qry[i])
                qry_losses.append(qry_loss.detach())
                qry_acc = (qry_logits.argmax(
                    dim=1) == y_qry[i]).sum().item() / querysz
                qry_accs.append(qry_acc)

                # Update the model's meta-parameters to optimize the query
                # losses across all of the tasks sampled in this batch.
                # This unrolls through the gradient steps.
                qry_loss.backward()
        meta_opt.step()

        qry_losses = sum(qry_losses) / task_num
        qry_accs = 100. * sum(qry_accs) / task_num
        i = epoch + float(batch_idx) / n_train_iter
        iter_time = time.time() - start_time
        if batch_idx % 4 == 0:
            print(
                f'[Epoch {i:.2f}] Train Loss: {qry_losses:.2f} | Acc: {qry_accs:.2f} | Time: {iter_time:.2f}'
            )

        log.append({
            'epoch': i,
            'loss': qry_losses,
            'acc': qry_accs,
            'mode': 'train',
            'time': time.time(),
        })

def test_maml(db, net, device, lr_finetune, epoch, log):
    # Crucially in our testing procedure here, we do *not* fine-tune
    # the model during testing for simplicity.
    # Most research papers using MAML for this task do an extra
    # stage of fine-tuning here that should be added if you are
    # adapting this code for research.
    net.train()
    n_test_iter = db.x_test.shape[0] // db.batchsz

    qry_losses = []
    qry_accs = []

    for batch_idx in range(n_test_iter):
        x_spt, y_spt, x_qry, y_qry = db.next('test')


        task_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)

        # TODO: Maybe pull this out into a separate module so it
        # doesn't have to be duplicated between `train` and `test`?
        n_inner_iter = 5
        inner_opt = torch.optim.SGD(net.parameters(), lr=lr_finetune)

        for i in range(task_num):
            with higher.innerloop_ctx(net, inner_opt, track_higher_grads=False) as (fnet, diffopt):
                # Optimize the likelihood of the support set by taking
                # gradient steps w.r.t. the model's parameters.
                # This adapts the model's meta-parameters to the task.
                for _ in range(n_inner_iter):
                    spt_logits = fnet(x_spt[i])
                    spt_loss = F.cross_entropy(spt_logits, y_spt[i])
                    diffopt.step(spt_loss)

                # The query loss and acc induced by these parameters.
                qry_logits = fnet(x_qry[i]).detach()
                qry_loss = F.cross_entropy(
                    qry_logits, y_qry[i], reduction='none')
                qry_losses.append(qry_loss.detach())
                qry_accs.append(
                    (qry_logits.argmax(dim=1) == y_qry[i]).detach())

    qry_losses = torch.cat(qry_losses).mean().item()
    qry_accs = 100. * torch.cat(qry_accs).float().mean().item()
    print(
        f'[Epoch {epoch+1:.2f}] Test Loss: {qry_losses:.2f} | Acc: {qry_accs:.2f}'
    )
    log.append({
        'epoch': epoch + 1,
        'loss': qry_losses,
        'acc': qry_accs,
        'mode': 'test',
        'time': time.time(),
    })

def test_sgd(db, net, device, epoch, log):
    print('testing')
    # Crucially in our testing procedure here, we do *not* fine-tune
    # the model during testing for simplicity.
    # Most research papers using MAML for this task do an extra
    # stage of fine-tuning here that should be added if you are
    # adapting this code for research.
    net.train()
    n_test_iter = db.x_test.shape[0] // db.batchsz

    qry_losses = []
    qry_accs = []

    for batch_idx in range(n_test_iter):
        x_spt, y_spt, x_qry, y_qry = db.next('test')


        task_num, setsz, c_, h, w = x_spt.size()

        for i in range(task_num):
            with torch.no_grad():
                # The query loss and acc induced by these parameters.
                qry_logits = net(x_qry[i]).detach()
                qry_loss = F.cross_entropy(
                    qry_logits, y_qry[i], reduction='none')
                qry_losses.append(qry_loss.detach())
                qry_acc =  (qry_logits.argmax(dim=1) == y_qry[i]).detach()
                qry_accs.append(qry_acc)

    qry_losses = torch.cat(qry_losses).mean().item()
    qry_accs = 100. * torch.cat(qry_accs).float().mean().item()
    print(
        f'[Epoch {epoch+1:.5f}] Test Loss: {qry_losses:.5f} | Acc: {qry_accs:.5f}'
    )
    log.append({
        'epoch': epoch + 1,
        'loss': qry_losses,
        'acc': qry_accs,
        'mode': 'test',
        'time': time.time(),
    })