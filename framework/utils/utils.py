import torch
from torch import nn
from torchvision import transforms
from framework.model.base import ResNetBackbone, FPNHead, MainModel
from PIL import Image
import os


def logging(path, flag='a+', message='log'):
    if os.path.isfile(path):
        with open(path, flag) as f:
            f.write(message)
    else:
        with open(path, 'w') as f:
            f.write(message)


def move(img):
    model = ResNetBackbone()
    features = model.get_features(img)
    fpn = FPNHead([img.shape[2:], features[0].shape[2:],
                   features[1].shape[2:], features[2].shape[2:]], same_size_output=True)

    res = fpn(features)
    return res


def get_shape_fpn(like_image_path, with_same_size=False):

    img = Image.open(like_image_path)
    inp_trans = transforms.Compose([transforms.ToTensor()])
    tensor_img = inp_trans(img)[None, :, :, :]
    model = ResNetBackbone()
    features = model.get_features(tensor_img)

    if with_same_size:
        return [tensor_img.shape[2:], features[0].shape[2:], features[1].shape[2:], features[2].shape[2:]]
    else:
        return [features[0].shape[2:], features[1].shape[2:], features[2].shape[2:]]
