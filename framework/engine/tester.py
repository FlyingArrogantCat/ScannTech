import torch
import torchvision
import json
import cv2
from torchvision import transforms
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
import time
import numpy as np
import matplotlib.pyplot as plt

from framework.model.base import MainModel
from framework.utils.utils import logging
from framework.data.dataset import MainDataset


class Tester:
    def __init__(self, ex_img_path, model_weigths_path, log_path, dataset_path, detach=True, lr=1e-3, device='cpu'):

        self.device = device
        self.log_path = log_path
        self.weigths_path = model_weigths_path

        sizes = get_size_from_backbone(ex_img_path)

        self.model = MainModel(size=sizes, same_size_output=True)
        self.model = self.model.to(self.device)

        self.optimizer = None
        self.detach = detach
        if not self.detach:
            self.optimizer = torch.optim.Adam([{'params': model.depthhead.parameters(), 'lr': lr},
                                               {'params': model.head.parameters(), 'lr': lr},
                                               {'params': model.backbone.parameters(), 'lr': lr * 1e-2}], lr=lr)

        self.loss_function = nn.MSELoss()
        self.dataset = MainDataset(dataset_path, split='val')
        self.dataloader = DataLoader(self.dataset)

    def testing(self):
        tt = time.clock()
        if self.optimizer is not None:
            self.optimizer.zero_grad()

        test_loss = 0
        test_loss_batch = []

        for indx, batch_sample in enumerate(self.dataloader):
            output = self.model(batch_sample['image'].to(self.device))

            if not self.detach:
                loss.backward()
                self.optimizer.step()

        print('Time: ', time.clock() - tt)

    def update_model(self, image_path):
        sizes = get_size_from_backbone(image_path)
        print(sizes)
        self.model = MainModel(size=sizes, same_size_output=True)
        self.model.load_state_dict(torch.load(self.weigths_path, map_location=torch.device('cpu')))
        self.model.eval()

    def test_sample(self, image_path, img_save_path=None):

        get_size_from_backbone
        sample = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
        input_transform = torchvision.transforms.Compose([transforms.ToTensor()])
        img = input_transform(sample)

        out = self.model(img[None, :, :, :]).detach().numpy()[0][0] * 255 * 2

        if img_save_path is not None and type(img_save_path) == str:
            cv2.imwrite(img_save_path, out)


def get_size_from_backbone(img_path):
    sizes = []

    model = MainModel(size=[0, 0, 0, 0])

    input_transform = torchvision.transforms.Compose([transforms.ToTensor()])
    image = Image.open(img_path)
    img = input_transform(image)

    c1, c2, c3, c4 = model.backbone.get_features(img[None, :, :, :])

    sizes.append(img.shape[1:3])
    sizes.append(c1.shape[2:4])
    sizes.append(c2.shape[2:4])
    sizes.append(c3.shape[2:4])
    sizes.append(c4.shape[2:4])

    return sizes
