from torch import nn
from Utils.Initializers import *
import torch.nn.functional as F

import math


class ClassificationNet(nn.Module):
    # Abstract class for classification models implementing cross-entropy loss and relative accuracy.
    def __init__(self, dev, loss_fn=nn.CrossEntropyLoss(), lr=0.001):
        super().__init__()

        self.lr = lr
        self.dev = dev
        self.loss_fn = loss_fn

    def loss(self, batch):
        x, y = batch
        l = self.loss_fn(self.forward(x), y.to(self.dev))
        return l

    def test_loss(self, batch):
        x, y = batch
        n_t = x.shape[0]
        output = self.forward(x)
        l = self.loss_fn(output, y.to(self.dev))
        predicted = torch.max(output, dim=1).indices
        correct_count = (predicted == y.to(self.dev)).sum()
        return l, correct_count / n_t


class ResidualBlock(nn.Module):
    """
    A residual block as defined by He et al.
    Taken from https://github.com/matthias-wright/cifar10-resnet
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super(ResidualBlock, self).__init__()
        self.conv_res1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   padding=padding, stride=stride, bias=False)
        self.conv_res1_bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.9)
        self.conv_res2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   padding=padding, bias=False)
        self.conv_res2_bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.9)

        if stride != 1:
            # in case stride is not set to 1, we need to downsample the residual so that
            # the dimensions are the same when we add them together
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_features=out_channels, momentum=0.9)
            )
        else:
            self.downsample = None

        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x

        out = self.relu(self.conv_res1_bn(self.conv_res1(x)))
        out = self.conv_res2_bn(self.conv_res2(out))

        if self.downsample is not None:
            residual = self.downsample(residual)

        out = self.relu(out)
        out = out + residual
        return out


class ResNet(ClassificationNet):
    """
    A Residual network, architecture taken from https://github.com/matthias-wright/cifar10-resnet
    """

    def __init__(self, dev, loss_fn=nn.CrossEntropyLoss()):
        super(ResNet, self).__init__(dev, loss_fn)

        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64, momentum=0.9),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=128, momentum=0.9),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(128, 128, 3, stride=1, padding=1),
            nn.Conv2d(128, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256, momentum=0.9),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256, momentum=0.9),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(256, 256, 3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc = nn.Linear(in_features=1024, out_features=10, bias=True)

    def forward(self, x):
        x = x.to(self.dev)
        out = self.conv(x)
        out = out.view(-1, out.shape[1] * out.shape[2] * out.shape[3])
        out = self.fc(out)
        return out
