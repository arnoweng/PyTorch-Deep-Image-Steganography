# encoding: utf-8


from torch.utils.data import Dataset
from skimage import io
from PIL import Image
import os


class PreProcessedDataSet(Dataset):
    def __init__(self, data_dir, image_list_file, secret_image_path="../secretImg/test.jpg", transform=None):
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
                label = -1  # 人脸数据集没有label
                image_name = os.path.join(data_dir, image_name)
                image_names.append(image_name)
                labels.append(label)

        self.image_names = image_names
        self.labels = labels
        self.transform = transform
        self.secret_image = Image.open(secret_image_path).convert('RGB')

        if self.transform is not None:
            self.secret_image = self.transform(self.secret_image)

    def __getitem__(self, index):
        """
        :param index: The ind/ex of item.
        :return: Image and its label.
        """
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = image  # label即为原始图片
        if self.transform is not None:
            image = self.transform(image)
            label = image  #label即为原始图片
            image = torch

        return image, label

    def __len__(self):
        return len(self.image_names)
