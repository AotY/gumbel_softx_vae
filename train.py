#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""

"""

from __future__ import print_function
import argparse
import numpy as np

import torch
from torch import optim
import torch.nn.functional as F

from torchvision import datasets, transforms
from torchvision.utils import save_image

from model import GumbelVAE

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--temp', type=float, default=1.0, metavar='S',
                    help='tau(temperature) (default: 1.0)')
parser.add_argument('--device', type=str, default='cuda', help='cpu or cuda')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')


latent_size = 20
category_size = 10 # one-hot vector
temperature_min = 0.5
ANNEAL_RATE = 0.00003

opt = parser.parse_args()

device =torch.device(opt.device)

model = GumbelVAE(
    784,
    latent_size,
    category_size,
    device)

optimizer = optim.Adam(model.parameters(), lr=0.001)

kwargs = {
    'num_workers': 2,
    'pin_memory': True
}

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', trian=True, download=True, transform=transforms.toTensor()),
    batch_size=opt.batch_size,
    shuffle=True,
    **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.toTensor()),
    batch_size=opt.batch_size,
    shuffle=True,
    **kwargs)

# reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_output, input, encode_output):
    BCE = F.binary_cross_entropy(recon_output, input.view(-1, 784), size_average=False)

    log_encode_output = torch.log(encode_output + 1e-20)
    g = torch.log(torch.Tensor([1.0 / category_size]), device=device)
    KLD = torch.sum(encode_output * (log_encode_output - g), dim=-1).mean()

    return BCE + KLD


def train(epoch):
    model.train()

    train_loss = 0
    temperature = opt.temperature
    for bi, (data, _) in enumerate(train_loader):
        data = data.to(device)

        optimizer.zero_grad()

        recon_output, encode_output = model(data, temperature)

        loss = loss_function(recon_output, data, encode_output)

        loss.backward()

        train_loss += loss[0].item()

        optimizer.step()

        if (bi + 1) % 100 == 0:
            temperature = np.maximun(temperature * np.exp(-ANNEAL_RATE * bi),
                                     temperature_min)

        if (bi + 1) % opt.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, bi * len(data), len(train_loader.dataset),
                100. * bi / len(train_loader),
                loss.data[0] / len(data)))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    temperature = opt.temperature
    with torch.no_grad():
        for bi, (data, _) in enumerate(test_loader):
            data.to(device)

            recon_output, encode_output = (data, temperature)
            test_loss += loss_function(recon_output, data, encode_output)[0].item()

            if (bi + 1) % 100 == 0:
                temperature = np.maximun(temperature * np.exp(-ANNEAL_RATE * bi),
                                         temperature_min)

            if (bi + 1) % 100 == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                    recon_output.view(opt.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.data.cpu(),
                        'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

def run():
    for epoch in range(1, opt.epochs + 1):
        train(epoch)
        test(epoch)

        sample = torch.randn(64, 200, device=device)

        sample = model.decode(sample).cpu
        save_image(sample.data.view(64, 1, 28, 28),
                   'results/sample_' + str(epoch) + '.png')


if __name__ == '__main__':
    run()

