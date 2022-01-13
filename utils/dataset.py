import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


from utils.utils import read_data


class DatasetBuilder(Dataset):

    def __init__(self, data_path, img_size, channels=3):
        self.data = read_data(data_path)
        self.img_size = img_size
        self.channels = channels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img_path = self.data[item]
        image = self.read_image(img_path)
        image = self.preprocess_image(image)
        return image

    def read_image(self, img_path):
        try:
            if self.channels == 1:
                image = cv.imread(img_path, 0)
            else:
                image = cv.imread(img_path)
        except:
            image = None
            print(f"ERROR: Can not read image: {img_path}")
            exit(1)
        return image

    def preprocess_image(self, image):
        if self.channels == 3:
            image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        image = image / 128. - 1.
        # Pad image if dimensions of the image are smaller than provided image size
        if self.channels == 1:
            image = np.expand_dims(image, 2)
        org_height, org_width, _ = image.shape
        if org_height < self.img_size[1]:
            pad_val = [(int((self.img_size[1] - org_height) / 2.),
                        int((self.img_size[1] - org_height) / 2.)),
                       (int((self.img_size[0] - org_width) / 2.),
                        int((self.img_size[0] - org_width) / 2.)),
                       (0, 0)]
            image = np.pad(image, pad_val)
        image = cv.resize(image, tuple(self.img_size))
        if self.channels == 1:
            image = np.expand_dims(image, 2)
        image = np.transpose(image, (2, 0, 1))
        return torch.FloatTensor(image)


def create_dataloader(data_path, img_size, batch_size, channels=3):
    """
    :param data_path: Path to json file with list of images
    :param img_size: Input size of the image - input to the discriminator net
    :param batch_size: Batch size
    :param channels: Number of channels of the input image
    :return: train_dataloader
    """
    train_data = DatasetBuilder(data_path, img_size, channels)
    train_dataloader = DataLoader(dataset=train_data, pin_memory=True, batch_size=batch_size, shuffle=True)
    return train_dataloader
