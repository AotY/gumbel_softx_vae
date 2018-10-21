#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""

"""

import torch
import torch.nn as nn
import torch.funtional as F

from gumbel_softmax import GumbelSoftmax


class GumbelVAE(nn.Module):
    def __init__(self,
                 input_size=784,
                 latent_size,
                 category_size,
                 device):
        super(GumbelVAE, self).__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.category_size = category_size

        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, latent_size * category_size)

        sefl.fc4 = nn.Linear(latent_size * category_size, 256)
        self.fc5 = nn.Linear(256, 512)
        self.fc6 = nn.Linear(512, input_sizse)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid(dim=2)

        self.gumbel_softmax = GumbelSoftmax(dim=2, 
                                            device=device)

    def encode(self, input):
        h1 = self.relu(self.fc1(input))
        h2 = self.relu(self.fc2(h1))
        h3 = self.relu(self.fc3(h2))

        return h3

    def decode(self, encode_output):
        h4 = self.relu(self.fc4(encode_output))
        h5 = self.relu(self.fc5(h4))
        output = self.sigmoid(self.fc6(h5))
        return output
    
    def forward(self, input, temperature):
        encode_output = self.encode(input)

        tmp = encode_output.view(encode_output.size(0),
                           self.latent_size,
                           self.category_size)

        tmp = self.gumbel_softmax(tmp, temperature)
        tmp = tmp.view(-1, slef.latent_size * self.category_size)

        decode_output = self.decode(tmp_softmax)
        return decode_output, F.softmax(encode_output)


