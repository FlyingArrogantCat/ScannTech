from torch import nn
from pathlib import Path
import cv2
from PIL import Image


class LidarLoader(nn.Module):
    def __init__(self, image_path, lidar_data_path):
        super(LidarLoader, self).__init__()

        self.image_path = Path(image_path)
        self.lidar_data_path = Path(lidar_data_path)

        assert self.image_path.is_dir() and self.lidar_data_path.is_dir(), "The input sting is not paths"

        self.list_image = [x for x in self.image_path.iterdir() if '.bmp' in str(x)]
        self.list_lidar_data = [x for x in self.lidar_data_path.iterdir() if '.txt' in str(x)]

        assert len(self.list_image) == len(self.list_lidar_data), "The input folders contain different number of files"


    def __getitem__(self, item):
        img = Image.open(str(self.list_image[item]))

        lidar_data = []
        with open(str(self.list_lidar_data[item]), 'r') as f:
            for line in f:
                line = line.replace('[', '').replace(']', '').split(',')
                line = [float(x.replace(' ', '')) for x in line]
                lidar_data.append(line)

        return img, lidar_data
