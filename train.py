import torch
from model.base import MainModel
from data.dataset import MainDataset
from engine.engine import Trainer

if __name__ == '__main__':
    split = 'train'
    sizes = []
    dataset = MainDataset('.\data', split)

    model = MainModel(size=[0, 0, 0, 0])
    sample = dataset.__getitem__(0)
    img = sample['image']
    c1, c2, c3, c4 = model.backbone.get_features(img[None, :, :, :])

    sizes.append(img.shape[1:3])
    sizes.append(c1.shape[2:4])
    sizes.append(c2.shape[2:4])
    sizes.append(c3.shape[2:4])
    sizes.append(c4.shape[2:4])

    model = MainModel(size=sizes, same_size_output=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=10e-3)

    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    Trainer = Trainer(model, dataset, optimizer, device=device, batch_size=1)

    Trainer.training(5)
