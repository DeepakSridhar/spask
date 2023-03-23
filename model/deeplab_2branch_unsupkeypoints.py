"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

This file incorporates work covered by the following copyright and permission notice:

	Copyright (c) 2018 Zilong Huang

	Permission is hereby granted, free of charge, to any person obtaining a copy
	of this software and associated documentation files (the "Software"), to deal
	in the Software without restriction, including without limitation the rights
	to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
	copies of the Software, and to permit persons to whom the Software is
	furnished to do so, subject to the following conditions:

	The above copyright notice and this permission notice shall be included in all
	copies or substantial portions of the Software.

	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
	AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
	LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
	OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
	SOFTWARE.

	source: https://github.com/speedinghzl/Pytorch-Deeplab/
"""

import numpy as np
import torch.nn as nn
import torch
from .imm_model import AssembleNet, get_gaussian_maps, get_coord
import utils.utils as utils

affine_par = True


def outS(i):
    i = int(i)
    i = (i + 1) / 2
    i = int(np.ceil((i + 1) / 2.0))
    i = (i + 1) / 2
    return i


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=1, stride=stride, bias=False)  # change
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False

        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,  # change
                               padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Classifier_Module(nn.Module):

    def __init__(self, dilation_series, padding_series, num_classes, dim=1024):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(nn.Conv2d(
                dim, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes, gauss_std=0.1, gauss_mode='ankush', args=None):
        self.inplanes = 64
        n_maps = num_classes
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=1, dilation=4)
        # self.layer5 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [
        #                                     6, 12, 18, 24], num_classes, 2048)
        

        # if args is not None:
        #     self.map_sizes = [int(args.input_size.split(',')[0])//8 + 1]
        #     sizes = [int(args.input_size.split(',')[0]), int(args.input_size.split(',')[0])]
        # else:
        #     self.map_sizes = [17]
        #     sizes = [128, 128]
        # print(sizes, self.map_sizes)
        # self.pose_net = AssembleNet(n_maps=n_maps, max_size=sizes, min_size=[16, 16])
        
        if args.use_mlp:
            self.ll = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.ReLU(True),
                nn.Dropout(0.05),
                nn.Linear(1024, 2),
            )
        else:
            self.ll = nn.Linear(2048, 2)


        self.gauss_std = gauss_std
        self.gauss_mode = gauss_mode

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par))
        layers = []
        layers.append(block(self.inplanes, planes, stride,
                            dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def _make_pred_layer(self, block, dilation_series, padding_series, num_classes, dim):
        return block(dilation_series, padding_series, num_classes, dim)

    def forward_once(self, x):
        inp = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        feat1 = x
        x = self.layer1(x)
        feat2 = x
        x = self.layer2(x)
        feat3 = x
        x = self.layer3(x)
        feat4 = x
        x4 = self.layer4(x)
        # split x4 in half
        c_n = int(x4.shape[1] / 2)
        x4_first = x4[:, :c_n, :, :]
        x4_second = x4[:, c_n:, :, :]

        feature_instance = x4_first
        feature_part = x4_second

        x5 = self.layer5(x4)

        gauss_pt = utils.batch_get_centers(nn.Softmax(dim=1)(x5), yx=True)

        
        gauss_xy = []
        for shape_hw in self.map_sizes:
            gauss_xy_hw = \
                get_gaussian_maps(gauss_pt, [shape_hw, shape_hw], 1.0 / self.gauss_std, mode=self.gauss_mode)
            gauss_xy.append(gauss_xy_hw)
        pose_embeddings = gauss_xy[-1]

        outputs = {'feature_instance': feature_instance,
        'feature': feature_part,
        'x5': x5,
        'gauss_pt': gauss_pt,
        'pose_embeddings': pose_embeddings,
        'x4': x4,
        }

        return outputs, feature_part, x5

    def forward(self, x, x2=None, lm=None):
        outputs, feature_part, xout = self.forward_once(x)
        if x2 is not None:
            outputs2, feature_part2, x2out = self.forward_once(x2)    
            pose_embeddings_tps = outputs2['pose_embeddings']
        else:
            pose_embeddings_tps = outputs['pose_embeddings']   

        if lm is not None:
            pose_embeddings_tps = lm   
            outputs['pose_embeddings'] = lm 

        future_im_pred = self.pose_net(outputs['x4'], pose_embeddings_tps)
        outputs['future_im_pred'] = future_im_pred

        return outputs, feature_part, xout


    def get_1x_lr_params_NOscale(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = []

        b.append(self.conv1)
        b.append(self.bn1)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        b.append(self.layer5.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i
    
    def get_100x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        b.append(self.pose_net.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters_linear(self, args):
        for name, param in self.named_parameters():
            # print(name)
            if args.optimize_bias:
                if 'bias' in name:
                    param.requires_grad = True
            else:
                if 'll' not in name:
                    param.requires_grad = False

        return [{'params':self.ll.parameters()}]

    def optim_parameters_all(self, args):

        return [{'params':self.parameters()}]

    def optim_parameters(self, args):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': args.learning_rate},
                {'params': self.get_10x_lr_params(), 'lr': 1 * args.learning_rate},
                {'params': self.get_100x_lr_params(), 'lr': 1 * args.learning_rate}]


def Res101_Deeplab_2branch_Geometric(args, num_classes=21):
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes, args=args)
    return model


def Res50_Deeplab_2branch_Geometric(args, num_classes=21):
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes, args=args)
    return model
