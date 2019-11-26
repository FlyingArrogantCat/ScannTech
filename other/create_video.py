import os
import cv2
from PIL import Image
import numpy as np
import pandas as pd


def generate_video():
    os.chdir("/home/fedor/projects/ScannTech/")
    path = "/home/fedor/projects/ScannTech/vidoe"

    video_name = 'video.avi'

    images = np.sort([img for img in os.listdir(path)])

    df = pd.DataFrame({'img': images, 'ind': [int(x[:len(x) - 4]) for x in images]})
    images = df.sort_values(by='ind')['img'].values

    frame = cv2.imread(os.path.join(path, images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(path, image)))
    video.release()

generate_video()
