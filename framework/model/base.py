import torch
import torchvision
from torch import nn as nn
from torch.nn import Module


class MainModel(Module):
    def __init__(self, size, same_size_output=False):
        super(MainModel, self).__init__()

        self.backbone = ResNetBackbone()
        self.head = FPNHead(size, same_size_output=same_size_output)
        self.depthhead = ConvBlock(256, 1, 1, 3)

    def forward(self, image):
        x = self.backbone.get_features(image)
        out = self.head(x)
        out = self.depthhead(out)
        return out


class ResNetBackbone(Module):
    def __init__(self):
        super(ResNetBackbone, self).__init__()
        self.net = torchvision.models.resnet18(pretrained=True)

    def get_features(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)
        c1 = self.net.layer1(x)
        c2 = self.net.layer2(c1)
        c3 = self.net.layer3(c2)
        c4 = self.net.layer4(c3)

        return c1, c2, c3, c4

    def forward(self, x):
        out = self.net.forward(x)
        return out


class FPNHead(Module):
    def __init__(self, size, same_size_output=False):
        super(FPNHead, self).__init__()
        self.same_size_output = same_size_output
        self.size = size
        self.block4 = ConvBlock(in_channels=512, out_channels=256, kernel_size=1, num_layer=1)
        self.block3 = ConvBlock(in_channels=256, out_channels=256, kernel_size=1, num_layer=1)
        self.block2 = ConvBlock(in_channels=128, out_channels=256, kernel_size=1, num_layer=1)
        self.block1 = ConvBlock(in_channels=64, out_channels=256, kernel_size=1, num_layer=1)

        self.define_upsampling(self)

    def forward(self, features):
        c4 = self.block4(features[3])
        c3 = self.upsample4(c4) + self.block3(features[2])
        c2 = self.upsample3(c3) + self.block2(features[1])
        c1 = self.upsample2(c2) + self.block1(features[0])

        if self.same_size_output:
            c1 = self.upsample1(c1)

        return c1

    def define_upsampling(self):
        if self.same_size_output:
            self.upsample4 = nn.Upsample(size=self.size[3], mode='bilinear', align_corners=True)
            self.upsample3 = nn.Upsample(size=self.size[2], mode='bilinear', align_corners=True)
            self.upsample2 = nn.Upsample(size=self.size[1], mode='bilinear', align_corners=True)
            self.upsample1 = nn.Upsample(size=self.size[0], mode='bilinear', align_corners=True)
        else:
            self.upsample4 = nn.Upsample(size=self.size[2], mode='bilinear', align_corners=True)
            self.upsample3 = nn.Upsample(size=self.size[1], mode='bilinear', align_corners=True)
            self.upsample2 = nn.Upsample(size=self.size[0], mode='bilinear', align_corners=True)

    def change_size(self, sizes):
        self.size = sizes
        self.define_upsampling(self)


class ConvBlock(Module):
    def __init__(self, in_channels=128, out_channels=256,
                 kernel_size=1, num_layer=1):
        super(ConvBlock, self).__init__()

        self.block = nn.Sequential()

        self.block.add_module('conv_1', nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels, kernel_size=kernel_size))
        self.block.add_module('bn_1', nn.BatchNorm2d(out_channels))
        self.block.add_module('relu_1', nn.ReLU())

        for i in range(1, num_layer):
            self.block.add_module(f'conv_{i+1}', nn.Conv2d(in_channels=out_channels,
                                  out_channels=out_channels, kernel_size=kernel_size))

            self.block.add_module(f'bn_{i+1}', nn.BatchNorm2d(out_channels))
            self.add_module(f'relu_{i+1}', nn.ReLU())

    def forward(self, x):
        out = self.block(x)
        return out
