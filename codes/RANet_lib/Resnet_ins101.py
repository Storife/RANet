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
# from resnet_ins import *

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


def get_imagenet_model(type, pretrained=True):
    if type == 'vgg16':
        return models.__dict__['vgg16'](pretrained=pretrained)
    if type == 'resnet50':
        return models.__dict__['resnet50'](pretrained=pretrained)
    if type == 'resnet101':
        return models.__dict__['resnet101'](pretrained=pretrained)
    if type == 'resnet_ins101':
        return models.__dict__['resnet_ins101'](pretrained=pretrained)
    if type == 'resnet_ins_RGBM101':
        return models.__dict__['resnet_ins_RGBM101'](pretrained=pretrained)
    else:
        print('avilible types: ')
        print(model_names)
        assert 'error model type'


