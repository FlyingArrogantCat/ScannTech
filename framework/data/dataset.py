from torchvision import transforms
from pathlib import Path
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image


class MainDataset(Dataset):
    def __init__(self, dataset_path, split='train', transform=None, normalize=False):
        super(MainDataset, self).__init__()
        self.normalize = None

        self.dataset_path = dataset_path
        self.split = split

        path = Path(self.dataset_path) / Path(str(self.split + '.nyu'))
        with open(str(path), 'r') as f:
            self.list_set = f.read()

        self.list_set = self._splitting(self, set=self.list_set)

        if transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])
        if normalize:
            self.normalize = transforms.Compose([transforms.Normalize([.485, .456, .406], [.229, .224, .225])])

    def __getitem__(self, indx):
        sample = self.list_set.iloc[indx]

        image = Image.open(str(self.dataset_path / Path('nyud') / Path(sample['image'])))
        label = Image.open(str(self.dataset_path / Path('nyud') / Path(sample['label'])))

        image = self.transform(image)
        label = self.transform(label)

        if self.normalize is not None:
            image = self.normalize(image)

        return {'image': image, 'label': label}

    def __len__(self):
        return len(self.list_set)

    @staticmethod
    def _splitting(self, set):
        names = set.split('\n')

        images = []
        labels = []
        for name in names:
            if name == '':
                continue
            paths = name.split('\t')
            images.append(paths[0])
            labels.append(paths[1])

        return pd.DataFrame({'image': images, 'label': labels})
