from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np
    
# def convbn_dw(in_planes, out_planes, kernel_size=3, kernels_per_layer=1):
#     return nn.sequential(nn.Conv2d(in_planes, in_planes * kernels_per_layer, kernel_size=kernel_size, padding=1, groups=in_planes),
#                          nn.Conv2d(in_planes * kernels_per_layer, out_planes, kernel_size=1),
#                          nn.BatchNorm2d(out_planes))

def BSConvS(in_planes, out_planes, kernel_size, stride, padding = 0, dilation = 1, p = 0.25, min_mid_channels = 4, padding_mode="zeros"):
    assert 0.0 <= p <= 1.0
    mid_channels = min(in_planes, max(min_mid_channels, math.ceil(p * in_planes)))

    BSConvS_Layer = nn.Sequential(
        #pointwise 1 + batch normalization
        nn.Conv2d(
            in_channels = in_planes,
            out_channels = mid_channels,
            kernel_size = (1, 1),
            stride = 1,
            padding = 0,
            dilation = 1,
            groups = 1,
            bias = False
        ),
        nn.BatchNorm2d(mid_channels),

        #pointwise 2 + batch normalization
        nn.Conv2d(
            in_channels = mid_channels,
            out_channels = out_planes,
            kernel_size = (1, 1),
            stride = 1,
            padding = 0,
            dilation = 1,
            groups = 1,
            bias = False
        ),
        nn.BatchNorm2d(num_features = out_planes),

        #depthwise
        nn.Conv2d(
            in_channels = out_planes,
            out_channels = out_planes,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            dilation = dilation,
            groups = out_planes
        )
    )

    return BSConvS_Layer

def BSConvS_3d(in_planes, out_planes, kernel_size, stride, padding = 0, min_mid_channels = 4, p = 0.25):
    assert 0.0 <= p <= 1.0
    mid_channels = min(in_planes, max(min_mid_channels, math.ceil(p * in_planes)))

    BSConvs_3d_Layer = nn.Sequential(
        #pointwise 1 + batch normalization
        nn.Conv3d(
            in_channels = in_planes,
            out_channels = mid_channels,
            kernel_size = 1,
            stride = 1,
            padding = 0,
            dilation = 1,
            groups = 1,
            bias = False
        ),
        nn.BatchNorm3d(mid_channels),

        #pointwise 2 + batch normalization
        nn.Conv3d(
            in_channels = mid_channels,
            out_channels = out_planes,
            kernel_size = 1,
            stride = 1,
            padding = 0,
            dilation = 1,
            groups = 1,
            bias = False
        ),
        nn.BatchNorm3d(out_planes),

        #depthwise
        nn.Conv3d(
            in_channels = out_planes,
            out_channels = out_planes,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            groups = out_planes
        )
    )

    return BSConvs_3d_Layer

def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):

    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation = dilation, bias=False),
                         nn.BatchNorm2d(out_planes))


def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):

    return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride,bias=False),
                         nn.BatchNorm3d(out_planes))

class BasicBlock_BSConvS(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        if stride == 1:
            self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=True))
            self.conv2 = BSConvS(planes, planes, 3, 1, pad, dilation)
        else:
            self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=True))
            self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out

class disparityregression(nn.Module):
    def __init__(self, maxdisp):
        super(disparityregression, self).__init__()
        self.disp = torch.Tensor(np.reshape(np.array(range(maxdisp)),[1, maxdisp,1,1])).cuda()

    def forward(self, x):
        out = torch.sum(x*self.disp.data,1, keepdim=True)
        return out

class feature_extraction(nn.Module):
    def __init__(self):
        super(feature_extraction, self).__init__()
        #Feature Extraction CNN
        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1,1,1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2,1,1) 
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1,1,1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1,1,2)
        
        #SPP Module
        self.branch1 = nn.Sequential(nn.AvgPool2d((64, 64), stride=(64,64)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch2 = nn.Sequential(nn.AvgPool2d((32, 32), stride=(32,32)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch3 = nn.Sequential(nn.AvgPool2d((16, 16), stride=(16,16)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch4 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8,8)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.lastconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(128, 32, kernel_size=1, padding=0, stride = 1, bias=False))

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        #                                           block.expansion = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
           downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,1,None,pad,dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        #Feature Extraction CNN
        output      = self.firstconv(x)
        output      = self.layer1(output)
        output_raw  = self.layer2(output)
        output      = self.layer3(output_raw)
        output_skip = self.layer4(output)

        #SPP Module
        output_branch1 = self.branch1(output_skip)
        output_branch1 = F.upsample(output_branch1, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear')

        output_branch2 = self.branch2(output_skip)
        output_branch2 = F.upsample(output_branch2, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear')

        output_branch3 = self.branch3(output_skip)
        output_branch3 = F.upsample(output_branch3, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear')

        output_branch4 = self.branch4(output_skip)
        output_branch4 = F.upsample(output_branch4, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear')

        output_feature = torch.cat((output_raw, output_skip, output_branch4, output_branch3, output_branch2, output_branch1), 1)
        output_feature = self.lastconv(output_feature)

        return output_feature
