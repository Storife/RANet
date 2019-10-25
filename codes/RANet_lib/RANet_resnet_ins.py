# ************************************
# Author: Ziqin Wang
# Email: ziqin.wang.edu@gmail.com
# Github: https://github.com/Storife
# ************************************
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch

__all__ = ['ResNet_ins', 'resnet_ins101']


model_urls = {
    'resnet_ins18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet_ins34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet_ins50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet_ins101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet_ins152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnet_ins_RGBM101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock_ins(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock_ins, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.InstanceNorm2d(planes, affine=True, track_running_stats=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.InstanceNorm2d(planes, affine=True, track_running_stats=False)
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


class Bottleneck_ins(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, expansion=4):
        super(Bottleneck_ins, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.InstanceNorm2d(planes, affine=True, track_running_stats=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.InstanceNorm2d(planes, affine=True, track_running_stats=False)
        self.conv3 = nn.Conv2d(planes, planes * expansion, kernel_size=1, bias=False)
        self.bn3 = nn.InstanceNorm2d(planes * expansion, affine=True, track_running_stats=False)
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


class ResNet_ins(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet_ins, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.InstanceNorm2d(64, affine=True, track_running_stats=False)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # block.expansion = int(block.expansion / 2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm2d(planes * block.expansion, affine=True, track_running_stats=False),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, expansion=block.expansion))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, expansion=block.expansion))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnet_ins101(pretrained=False, **kwargs):
    """Constructs a ResNet_ins-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_ins(Bottleneck_ins, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet_ins101']), strict=False)
    return model
