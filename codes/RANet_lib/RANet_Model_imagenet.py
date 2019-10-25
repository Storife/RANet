# ************************************
# Author: Ziqin Wang
# Email: ziqin.wang.edu@gmail.com
# Github: https://github.com/Storife
# ************************************
import torch
import torch.nn as nn
from numpy.random import normal
from numpy.linalg import svd
from math import sqrt
import torchvision.models as models
from RANet_resnet_ins import *


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


def get_imagenet_model(type, pretrained=True):
    if type == 'resnet101':
        return models.__dict__['resnet101'](pretrained=pretrained)
    if type == 'resnet_ins101':
        return resnet_ins101(pretrained=pretrained)
    else:
        print('avilible types: ')
        print(model_names)
        assert 'error model type'


class ResNet101(nn.Module):
    def __init__(self, with_relu=0, pretrained=True, fc=False):
        super(ResNet101, self).__init__()
        self.with_relu = True
        self.base_model = get_imagenet_model('resnet_ins101', pretrained=pretrained)
        if not(fc):
            self.base_model._modules.pop('fc')

    def res_forward(self, x, fc=False):
        res_features = []
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        res_features.append(x)
        x = self.base_model.maxpool(x)

        x = self.base_model.layer1(x)
        res_features.append(x)
        x = self.base_model.layer2(x)
        res_features.append(x)
        x = self.base_model.layer3(x)
        res_features.append(x)
        x = self.base_model.layer4(x)
        res_features.append(x)
        if not(fc):
            return res_features
        x = self.base_model.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.base_model.fc(x)
        res_features.append(x)
        return res_features
    def res_forward_part(self, x, block):
        res_features = []
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        res_features.append(x)
        if block == 1:
            return res_features
        x = self.base_model.maxpool(x)

        x = self.base_model.layer1(x)
        res_features.append(x)
        if block == 2:
            return res_features
        x = self.base_model.layer2(x)
        res_features.append(x)
        if block == 3:
            return res_features
        x = self.base_model.layer3(x)
        res_features.append(x)
        if block == 4:
            return res_features
        x = self.base_model.layer4(x)
        res_features.append(x)
        if block == 5:
            return res_features

        x = self.base_model.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.base_model.fc(x)
        res_features.append(x)
        return res_features

    def res_forward_merge(self, x, in_fea, merge_layer, fc=False):
        res_features = []
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        res_features.append(x)
        x = self.base_model.maxpool(x)
        x = self.base_model.layer1(x)
        res_features.append(x)

        x = torch.cat([x, in_fea], 1)
        x = merge_layer(x)

        x = self.base_model.layer2(x)
        res_features.append(x)
        x = self.base_model.layer3(x)
        res_features.append(x)
        x = self.base_model.layer4(x)
        res_features.append(x)
        if not(fc):
            return res_features
        x = self.base_model.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.base_model.fc(x)
        res_features.append(x)
        return res_features

    def _initialize_weights(self):
        pass




