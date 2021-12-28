import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from utils.utils import read_data


class DatasetBuilder(Dataset):

    def __init__(self, data_path, img_size):
        self.data = read_data(data_path)
        self.img_size = img_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img_path = self.data[item]
        img = self.read_image(img_path)
        image = self.preprocess_image(img, img_path)
        return image

    def read_image(self, img_path):
        try:
            image = cv.imread(img_path)
        except:
            image = None
            print(f"ERROR: Can not read image: {img_path}")
            exit(1)
        return image

    def preprocess_image(self, image, img_path):
        try:
            image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
            image = cv.resize(image, tuple(self.img_size))
            image = np.transpose(image, (2, 0, 1))
            image = image / 255.
        except:
            print(f"ERROR: Can not pre-process image: {img_path}")
            exit(1)
        return torch.FloatTensor(image)


def create_dataloader(data_path, img_size, batch_size):
    train_data = DatasetBuilder(data_path, img_size)
    train_dataloader = DataLoader(dataset=train_data, pin_memory=True, batch_size=batch_size, shuffle=True)
    return train_dataloader
