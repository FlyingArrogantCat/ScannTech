import torch
from torch import nn
from torchvision import transforms
from model.base import ResNetBackbone, FPNHead, MainModel
from PIL import Image
from data.dataset import MainDataset
from torch.utils.data import DataLoader


class Trainer(nn.Module):
    def __init__(self, model, dataset, optimizer, device, batch_size=1):
        super(Trainer, self).__init__()
        self.dataset = dataset
        self.model = model
        self.device = device
        if self.device:
            self.model = self.model.to(self.device)

        self.batch_size = batch_size
        self.device = device
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        self.optimizer = optimizer

    def training(self, epoch):
        epoch_loss = 0
        self.optimizer.zero_grad()
        for indx, batch_sample in enumerate(DataLoader):
            output = self.model(batch_sample['image'])
            epoch_loss += self.loss_function(batch_sample['label'], output)

        epoch_loss.backward()
        self.optimizer.step()

        print(f'Epoch: {epoch}, loss:{epoch_loss.detach().cpu()}')
