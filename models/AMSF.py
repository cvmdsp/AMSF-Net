import torch
import torch.nn as nn
import torch.nn.functional as F
# # from .basic_blocks import *
# import numpy as np
# import cv2
# import math
# from models.Transformer import TransformerModel
from models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from mymodels import MFF

from functools import partial
nonlinearity = partial(F.relu, inplace=True)



# 通道注意力模块
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 4, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 4, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        out = x*self.sigmoid(out)
        return out


# 空间注意力模块
class SpatialAttention(nn.Module):
    def __init__(self,kernel_size=3):
        super(SpatialAttention, self).__init__()
        #
        # assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        # padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x0):
        avg_out = torch.mean(x0, dim=1, keepdim=True)
        max_out, _ = torch.max(x0, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        out = x0*self.sigmoid(x)
        return out

###################################
class PCP(nn.Module):
    def __init__(self, in_channels, BatchNorm,k,p,dilation,padding):   #padding[1,2,3,6]  dilation[1,2,3,6]
        super(PCP, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 1)
        self.bn1 = BatchNorm(in_channels)
        self.relu1 = nn.ReLU()
        ###############################################
        self.dilate1 = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, dilation=dilation[0], padding=padding[0])

        self.dilate2 = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, dilation=dilation[1], padding=padding[1])

        self.dilate3 = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, dilation=dilation[2], padding=padding[2])

        self.dilate4 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3,  dilation=dilation[3], padding=padding[3])
        )

        ################################################
        self.BSP1 = BSP(in_channels, in_channels,k,p)
        self.BSP2 = BSP(in_channels, in_channels,k,p)
        self.BSP3 = BSP(in_channels, in_channels,k,p)
        self.BSP4 = BSP(in_channels, in_channels,k,p)

        self.Conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            BatchNorm(in_channels),
            nn.ReLU()
        )
        self.Conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            BatchNorm(in_channels),
            nn.ReLU()
        )
        self.Conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            BatchNorm(in_channels),
            nn.ReLU()
        )
        self.Conv4 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            BatchNorm(in_channels),
            nn.ReLU()
        )

        self.Conv_out = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            BatchNorm(in_channels),
            nn.ReLU()
        )
        self._init_weight()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        #第一分支
        dilate1_out = nonlinearity(self.dilate1(x))
        Fea1 = self.BSP1(dilate1_out)
        F1 = self.Conv1(Fea1)
        #第二分支
        dilate2_out = nonlinearity(self.dilate2(x))
        # Fea2 = self.BSP2(torch.cat((dilate2_out, Fea1), dim=1))
        Fea2 = self.BSP2(dilate2_out+Fea1)
        F2 = self.Conv2(Fea2)
        #第三分支
        dilate3_out = nonlinearity(self.dilate3(x))
        # Fea3 = self.BSP3(torch.cat((dilate3_out, Fea2), dim=1))
        Fea3 = self.BSP3(dilate3_out+Fea2)
        F3 = self.Conv3(Fea3)
        # 第四分支
        dilate4_out = nonlinearity(self.dilate4(x))
        # Fea4 = self.BSP4(torch.cat((dilate4_out, Fea3), dim=1))
        Fea4 = self.BSP4(dilate4_out+Fea3)
        F4 = self.Conv4(Fea4)
        #汇总
        F = F1+F2+F3+F4
        out = self.Conv_out(F)
        return out

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class BSP(nn.Module):
    def __init__(self, in_channel, out_channel,k,p):
        super(BSP, self).__init__()

        self.deconv1 = nn.Conv2d(
            in_channel//2, out_channel//2, (1, k), padding=(0, p)   #(9,4)
        )
        self.deconv2 = nn.Conv2d(
            in_channel//2, out_channel//2, (k, 1), padding=(p, 0)
        )
        self.depthwise_conv1 = DepthWiseConv(in_channel=out_channel//2, out_channel=out_channel//2)
        self.depthwise_conv2 = DepthWiseConv(in_channel=out_channel//2, out_channel=out_channel//2)

        self.ca = ChannelAttention(out_channel)
    def forward(self, input):
        B, C, H, W = input.shape
        input_1, input_2 = torch.split(input, C//2, dim=1)
        x1 = self.deconv1(input_1)
        x2 = self.deconv2(input_2)
        x3 = self.depthwise_conv1(x1)
        x4 = self.depthwise_conv2(x2)
        x = torch.cat((x3, x4), 1)
        out = self.ca(x)

        return out + input





class DepthWiseConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        # 这一行千万不要忘记
        super(DepthWiseConv, self).__init__()

        # 逐通道卷积
        self.depth_conv = nn.Conv2d(in_channels=in_channel,
                                    out_channels=in_channel,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=in_channel)
        # groups是一个数，当groups=in_channel时,表示做逐通道卷积

        # 逐点卷积
        self.point_conv = nn.Conv2d(in_channels=in_channel,
                                    out_channels=out_channel,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out




#################################################################
class AMSF(nn.Module):
    def __init__(self,
                 scale_ratio,
                 n_select_bands,
                 n_bands
                 ):
        """Load the pretrained ResNet and replace top fc layer."""
        super(AMSF, self).__init__()

        self.scale_ratio = scale_ratio
        self.n_bands = n_bands
        self.n_select_bands = n_select_bands
        self.weight = nn.Parameter(torch.tensor([0.5]))
        dim =16
        self.conv_fus = nn.Sequential(
            nn.Conv2d(n_bands, n_bands, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.conv_spat = nn.Sequential(
            nn.Conv2d(n_bands, n_bands, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_bands, n_bands, kernel_size=3, stride=1, padding=1),
        )
        self.conv_spec = nn.Sequential(
            nn.Conv2d(n_bands, n_bands, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(n_select_bands, n_bands, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.U1 = nn.Sequential(             #Chikusei通道数改为128
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

        )

        self.U2 = nn.Sequential(            #Chikusei通道数改为256
            nn.Conv2d(192, 96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),


        )
        self.U3 = nn.Sequential(            #Chikusei通道数改为256
            nn.Conv2d(192, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(             #Chikusei通道数改为128
            nn.Conv2d(32, n_bands, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_bands, n_bands, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(             #Chikusei通道数改为128
            nn.Conv2d(32, n_bands, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_bands, n_bands, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.conv_n_select_bands = nn.Sequential(             #Chikusei通道数改为128
            nn.Conv2d(n_select_bands, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

        )
        self.conv_n_bands = nn.Sequential(
            nn.Conv2d(n_bands, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )


        self.MFF_1 = MFF.MFF1(32,32)            #Chikusei通道数改为128
        self.MFF_2 = MFF.MFF1(32,32)            #Chikusei通道数改为128
        self.MFF_3 = MFF.MFF1(32, 32)            #Chikusei通道数改为128


        self.att = nn.Sequential(ChannelAttention(32),
                                 SpatialAttention())
        self.PCP_1 = PCP(96, nn.BatchNorm2d,3,1,[1,2,1,2],[1,2,1,2])
        self.PCP_2 = PCP(192, nn.BatchNorm2d,9,4,[1,2,3,6],[1,2,3,6])
        self.PCP_3 = PCP(192, nn.BatchNorm2d,9,4,[1,2,3,6],[1,2,3,6])

        self.D1 = nn.Sequential(
            nn.Conv2d(n_select_bands, 24, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(24, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

        )
        self.D2 = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(48, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32,32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.D3 = nn.Sequential(
            nn.Conv2d(n_bands, 48, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(48, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

    def lrhr_interpolate(self, x_lr, x_hr):
        x_lr = F.interpolate(x_lr, scale_factor=self.scale_ratio, mode='bilinear')
        gap_bands = self.n_bands / (self.n_select_bands - 1.0)
        for i in range(0, self.n_select_bands - 1):
            x_lr[:, int(gap_bands * i), ::] = x_hr[:, i, ::]
        x_lr[:, int(self.n_bands - 1), ::] = x_hr[:, self.n_select_bands - 1, ::]
        return x_lr

    def spatial_edge(self, x):
        edge1 = x[:, :, 0:x.size(2) - 1, :] - x[:, :, 1:x.size(2), :]
        edge2 = x[:, :, :, 0:x.size(3) - 1] - x[:, :, :, 1:x.size(3)]

        return edge1, edge2

    def spectral_edge(self, x):
        edge = x[:, 0:x.size(1) - 1, :, :] - x[:, 1:x.size(1), :, :]

        return edge

    def forward(self, x_lr, x_hr):

        x_hr_nands = self.conv_n_select_bands(x_hr)
        a = F.interpolate(x_hr_nands, scale_factor=1 / 4, mode='bilinear')
        a = self.att(a)

        b = F.interpolate(a, scale_factor=1 / 4, mode='bilinear')
        b = self.att(b)

        x_lr_nands = self.conv_n_bands(x_lr)
        c = F.interpolate(x_lr_nands, scale_factor=1 / 4, mode='bilinear')
        c = self.att(c)

        d = F.interpolate(x_lr_nands, scale_factor=4, mode='bilinear')
        d = self.att(d)



        e = self.MFF_1(b, c)    #
        f1 = torch.cat((torch.cat((b, c), 1), e), 1)
        f1 = self.PCP_1(f1)    #96
        f1 = F.interpolate(f1, scale_factor=4, mode='bilinear')   #128
        f1 = self.U1(f1)
 ###################################################################################
        g = self.MFF_2(a, x_lr_nands)    #128
        f2 = torch.cat((torch.cat((a, self.conv_n_bands(x_lr)),1), g),1)
        f2 = torch.cat((f2, f1),1)  #48+16
        f2 = self.PCP_2(f2)
        f2 = F.interpolate(f2, scale_factor=4, mode='bilinear')    #512
        f2 = self.U2(f2)

        h = self.MFF_3(d, x_hr_nands)  # 128
        f3 = torch.cat((d, self.conv_n_select_bands(x_hr),h), 1)
        f3 = torch.cat((f3, f2), 1)
        f3 = self.PCP_3(f3)
        f3 = self.U3(f3)

        x = self.conv3(f3+d)
        x = x + self.conv_spat(x)
        spat_edge1, spat_edge2 = self.spatial_edge(x)
        spec_edge = self.spectral_edge(x)
        return x,  spat_edge1, spat_edge2, spec_edge