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

# These Omniglot loaders are from Jackie Loong's PyTorch MAML implementation:
#     https://github.com/dragen1860/MAML-Pytorch
#     https://github.com/dragen1860/MAML-Pytorch/blob/master/omniglot.py
#     https://github.com/dragen1860/MAML-Pytorch/blob/master/omniglotNShot.py

import  torchvision.transforms as transforms
from    PIL import Image
import  numpy as np

import torch
import  torch.utils.data as data
import  os
import  os.path
import  errno

class ToyNShot:

    def __init__(self, batchsz, k_shot, k_query, device=None):
        """
        Different from mnistNShot, the
        :param batchsz: task num
        :param k_shot:
        :param k_qry:
        """


        # self.x_train, self.x_test = self.x[:1200], self.x[1200:]

        x = np.load("./support/data/X.npy")
        y = np.load("./support/data/Y.npy")

        y = np.expand_dims(y, axis=3)

        self.x_train = np.array([x[0], x[2], x[3]])

        self.x_train_sp = torch.from_numpy(np.array([x[0, 0:1, 0:5], 
                                                     x[2, 0:1, 0:5],
                                                     x[3, 0:1, 0:5]], 
                                                     dtype=np.double)).double()
        self.x_train_q = torch.from_numpy(np.array([x[0, 0:1, 5:15], 
                                                    x[2, 0:1, 5:15], 
                                                    x[3, 0:1, 5:15]], dtype=np.double)).double()

        self.y_train_sp = torch.from_numpy(np.array([y[0, 0:1, 0:5], 
                                                     y[2, 0:1, 0:5], 
                                                     y[3, 0:1, 0:5]], dtype=np.double)).double()
        self.y_train_q = torch.from_numpy(np.array([y[0, 0:1, 5:15],
                                                    y[2, 0:1, 5:15], 
                                                    y[3, 0:1, 5:15]], dtype=np.double)).double()

        self.x_test_sp = torch.from_numpy(np.array([x[1, 0:1, 0:5], 
                                                    x[4, 0:1, 0:5]], dtype=np.double)).double()
        self.x_test_q = torch.from_numpy(np.array([x[1, 0:1, 5:15], 
                                                   x[4, 0:1, 5:15]], dtype=np.double)).double()

        self.y_test_sp = torch.from_numpy(np.array([y[1, 0:1, 0:5], 
                                                    y[4, 0:1, 0:5]], dtype=np.double)).double()
        self.y_test_q = torch.from_numpy(np.array([y[1, 0:1, 5:15], 
                                                    y[4, 0:1, 5:15]], dtype=np.double)).double()

        print("X train support set is ", self.x_train_sp.shape)
        print("Y train support set is ", self.y_train_sp.shape)

        self.batchsz = batchsz
        self.k_shot = k_shot  # k shot
        self.k_query = k_query  # k query
        assert (k_shot + k_query) <=20


    def next(self, mode='train'):
        """
        Gets next batch from the dataset with name.
        :param mode: The name of the splitting (one of "train", "val", "test")
        :return:
        """

        if mode == "train":
            yield from (self.x_train_sp, self.y_train_sp, self.x_train_q, self.y_train_q)

        else:
            yield from (self.x_test_sp, self.y_test_sp, self.x_test_q, self.y_test_q)

        # if self.indexes[mode] >= len(self.datasets_cache[mode]):
        #     self.indexes[mode] = 0
        #     self.datasets_cache[mode] = self.load_data_cache(self.datasets[mode])

        # next_batch = self.datasets_cache[mode][self.indexes[mode]]
        # self.indexes[mode] += 1

        # return next_batch
