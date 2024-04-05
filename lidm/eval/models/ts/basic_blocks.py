#!/usr/bin/env python
# encoding: utf-8
'''
@author: Xu Yan
@file: basic_blocks.py
@time: 2021/4/14 22:53
'''
import torch.nn as nn

try:
    import torchsparse.nn as spnn
except:
    print('To install torchsparse 1.4.0, please refer to https://github.com/mit-han-lab/torchsparse/tree/74099d10a51c71c14318bce63d6421f698b24f24')


class BasicConvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(
                inc,
                outc,
                kernel_size=ks,
                dilation=dilation,
                stride=stride), spnn.BatchNorm(outc),
            spnn.ReLU(True))

    def forward(self, x):
        out = self.net(x)
        return out


class BasicDeconvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(
                inc,
                outc,
                kernel_size=ks,
                stride=stride,
                transposed=True),
            spnn.BatchNorm(outc),
            spnn.ReLU(True))

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(
                inc,
                outc,
                kernel_size=ks,
                dilation=dilation,
                stride=stride), spnn.BatchNorm(outc),
            spnn.ReLU(True),
            spnn.Conv3d(
                outc,
                outc,
                kernel_size=ks,
                dilation=dilation,
                stride=1),
            spnn.BatchNorm(outc))

        self.downsample = nn.Sequential() if (inc == outc and stride == 1) else \
            nn.Sequential(
                spnn.Conv3d(inc, outc, kernel_size=1, dilation=1, stride=stride),
                spnn.BatchNorm(outc)
            )

        self.ReLU = spnn.ReLU(True)

    def forward(self, x):
        out = self.ReLU(self.net(x) + self.downsample(x))
        return out
