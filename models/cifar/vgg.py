#
# CIFAR VGG models.
#
# Hyperspherical Prototypical Networks.
#

import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

#
#
#
class CIFARvgg(nn.Module):
    
    #
    #
    #
    def __init__(self, features, output_dim):
        super(CIFARvgg, self).__init__()
        self.features = features

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, output_dim),
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    #
    #
    #
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

#
#
#
def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

#
#
#
cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
          512, 512, 512, 512, 'M'],
}


def CIFARvgg11(output_dim):
    """VGG 11-layer model (configuration "A")"""
    return CIFARvgg(make_layers(cfg['A']), output_dim)


def CIFARvgg11_bn(output_dim):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return CIFARvgg(make_layers(cfg['A'], batch_norm=True), output_dim)


def CIFARvgg13(output_dim):
    """VGG 13-layer model (configuration "B")"""
    return CIFARvgg(make_layers(cfg['B']), output_dim)


def CIFARvgg13_bn(output_dim):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return CIFARvgg(make_layers(cfg['B'], batch_norm=True), output_dim)


def CIFARvgg16(output_dim):
    """VGG 16-layer model (configuration "D")"""
    return CIFARvgg(make_layers(cfg['D']), output_dim)


def CIFARvgg16_bn(output_dim):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return CIFARvgg(make_layers(cfg['D'], batch_norm=True), output_dim)


def CIFARvgg19(output_dim):
    """VGG 19-layer model (configuration "E")"""
    return CIFARvgg(make_layers(cfg['E']), output_dim)


def CIFARvgg19_bn(output_dim):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return CIFARvgg(make_layers(cfg['E'], batch_norm=True), output_dim)
