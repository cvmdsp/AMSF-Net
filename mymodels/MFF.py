# -*- coding: utf-8 -*-
# Copyright 2022 ByteDance
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
from mymodels.DRSA import *


def _make_pair(value):
    if isinstance(value, int):
        value = (value,) * 2
    return value


def conv_layer(in_channels,
               out_channels,
               kernel_size,
               bias=True):
    """
    Re-write convolution layer for adaptive `padding`.
    """
    kernel_size = _make_pair(kernel_size)
    padding = (int((kernel_size[0] - 1) / 2),
               int((kernel_size[1] - 1) / 2))
    return nn.Conv2d(in_channels,
                     out_channels,
                     kernel_size,
                     padding=padding,
                     bias=bias)


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    """
    Activation functions for ['relu', 'lrelu', 'prelu'].

    Parameters
    ----------
    act_type: str
        one of ['relu', 'lrelu', 'prelu'].
    inplace: bool
        whether to use inplace operator.
    neg_slope: float
        slope of negative region for `lrelu` or `prelu`.
    n_prelu: int
        `num_parameters` for `prelu`.
    ----------
    """
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError(
            'activation layer [{:s}] is not found'.format(act_type))
    return layer


def sequential(*args):
    """
    Modules will be added to the a Sequential Container in the order they
    are passed.
    
    Parameters
    ----------
    args: Definition of Modules in order.
    -------

    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError(
                'sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)




class LFE(nn.Module):


    def __init__(self, esa_channels, n_feats):
        super(LFE, self).__init__()
        f = esa_channels
        self.conv0 = nn.Conv2d(n_feats, f, kernel_size=1)
        self.conv_f = nn.Conv2d(f, f, kernel_size=1)
        self.conv3 = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

        c1_r = conv_layer(n_feats, n_feats, 3)
        c2_r = conv_layer(n_feats, n_feats, 3)
        c3_r = conv_layer(n_feats, n_feats, 3)
        act = activation('lrelu', neg_slope=0.05)
        self.conv1 = sequential(c1_r, act, c2_r, act)  #, c3_r, act


    def forward(self, x):
        c1_ = self.conv0(x)   #32
        v_max = F.max_pool2d(c1_, kernel_size=2, stride=2)
        # v_max = F.interpolate(c1_, scale_factor=2, mode='bilinear')  # 128
        c3 = self.conv3(v_max)
        # c3 = F.max_pool2d(c3, kernel_size=2, stride=2)
        c3 = F.interpolate(c3, scale_factor=2, mode='bilinear')  # 128

        cf = self.conv4(c1_+c3)
        m = self.sigmoid(cf)

        x = self.conv1(x)
        out = x * m

        return out




class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()

        self.dim = dim
        self.norm = norm_layer(4*dim)

    def forward(self, x):
        B, C, H, W = x.shape
        # 在行和列方向上间隔1选取元素
        x0 = x[:, :, 0::2, 0::2]  # B C H/2 W/2
        x1 = x[:, :, 1::2, 0::2]  # B C H/2 W/2
        x2 = x[:, :, 0::2, 1::2]  # B C H/2 W/2
        x3 = x[:, :, 1::2, 1::2]  # B C H/2 W/2
        # 拼接到一起作为一整个张量
        x = torch.cat([x0, x1, x2, x3], 1)  # 4*C B H/2 W/2
        x = self.norm(x.permute(0,2,3,1))  # 归一化操作

        return x.permute(0,3,1,2)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, res_scale=1):
        super(ResBlock, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=True)
        self.relu = activation('lrelu', neg_slope=0.05)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=True)

    def forward(self, x):
        x1 = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out * self.res_scale + x1
        return out

class MFF1(nn.Module):
    """
    Residual Local Feature Block (RLFB).
    """

    def __init__(self,
                 in_channels,
                 esa_channels=16):
        super(MFF1, self).__init__()

        self.LFE1 = LFE(esa_channels, in_channels)
        self.LFE2 = LFE(esa_channels, in_channels)
        self.LFE3 = LFE(esa_channels, in_channels)
        self.LFE4 = LFE(esa_channels, in_channels)
        act = activation('lrelu', neg_slope=0.05)


        self.Resblock1 = sequential(
            nn.Conv2d(2*in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            act,
            ResBlock(in_channels, in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            act,
        )
        self.Resblock2 = sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            act,
            ResBlock(in_channels, in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            act,
        )
        self.c1 = sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            act
        )
        self.c2 = sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            act
        )
        self.c3 = sequential(
            nn.Conv2d(in_channels*3, in_channels, kernel_size=3, stride=1, padding=1),
            act
        )
        self.DRSA1 = DRSA(channel_num=in_channels, bias=True, ffn_bias=True, window_size=2, with_pe=False, dropout=0.1)
        # self.DRSA2 = DRSA(channel_num=in_channels, bias=True, ffn_bias=True, window_size=2, with_pe=False, dropout=0.1)




    def forward(self, x, y):
        # brand1
        x1 = self.LFE1(x)
        x2 = self.LFE3(self.c1(x1))
        # x2 = self.DRSA1(x2)

        # brand2
        y1 = self.LFE2(y)
        y2 = self.LFE4(self.c1(y1))
        # y2 = self.DRSA2(y2)

        #brand1+brand2
        z = torch.cat((x, y), dim= 1)
        z1 = self.Resblock1(z)
        z2 = self.Resblock2(z1+x1+y1)
        z3 = torch.cat((x2+x, y2+y, z2), dim= 1)
        z4 = self.c3(z3)
        out = self.DRSA1(z4)

        return out

