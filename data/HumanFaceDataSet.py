# encoding: utf-8

"""
@author: Arno Weng
@contact: wengxy@pku.edu.cn

@version: 1.0
@file: read_data.py
@time: 2017/10/24 22:08

Read data from the directory 'DataSet'.
"""

from torch.utils.data import Dataset
from skimage import io
from PIL import Image
import os


class HumanFaceDataSet(Dataset):
    def __init__(self, data_dir, image_list_file, transform=None):
        """
        :param data_dir: Path to data directory.
        :param image_list_file: Path to the file containing images
         with corresponding labels.
        transform (optional): Optional transform to be applied on
         a sample.
        """
        image_names = []
        labels = []
        with open(image_list_file, "r") as f:
            for line in f:
                items = line.split()
                image_name = items[0]
                label = -1  #人脸数据集没有label
                image_name = os.path.join(data_dir, image_name)
                image_names.append(image_name)
                labels.append(label)

        self.image_names = image_names
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        """
        :param index: The ind/ex of item.
        :return: Image and its label.
        """
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = -1     #人脸数据集没有label
        if self.transform is not None:
            image = self.transform(image)

        return image,label

    def __len__(self):
        return len(self.image_names)
