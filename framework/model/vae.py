import torch
import torchvision
import numpy as np
from torch import nn
from .base import ConvBlock


class VAE(nn.Module):
    def __init__(self, image_size):
        super.__init__(VAE, self)
        self.size = image_size
        self.encoder = SEncoder()
        self.decoder = SDecoder()

    def forward(self, item):
        self.encoder(item)
        self.decoder(item)

    def to_gpu(self):
        if torch.cuda.is_available():
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()


class SEncoder(nn.Module):
    def __init__(self):
        super.__init__(SEncoder, self)
        self.backbone = torchvision.models.resnet18(pretrained=True)

    def forward(self, item):
        result = self.backbone(item)
        return result

    def cuda(self):
        self.backbone = self.backbone.cuda()


class SDecoder(nn.Module):
    def __init__(self):
        super.__init__(SEncoder, self)
        self.relu = nn.ReLU()
        self.conv1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2)
        self.conv2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2)
        self.conv3 = nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=3, stride=2)
        self.conv4 = nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=3, stride=2)

    def cuda(self):
        RaiseNotImplementError
