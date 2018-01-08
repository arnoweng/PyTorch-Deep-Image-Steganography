
import torchvision.datasets as dset

import utils.transformed as transforms

from data.HumanFaceDataSet import HumanFaceDataSet

class Get_dataset():
    def __init__(self,dataset_name,imageSize,data_dir,image_list_file):
        self.dataset_name=dataset_name
        self.data_dir=data_dir
        self.imageSize=imageSize
        self.image_list_file=image_list_file



    #返回数据集
    def get_dataset(self):
        dataset = None
        if self.dataset_name in ['imagenet', 'folder', 'lfw']:
            # folder dataset
            dataset = dset.ImageFolder(root=self.data_dir,
                                       transform=transforms.Compose([
                                           transforms.Resize(self.imageSize),
                                           transforms.CenterCrop(self.imageSize),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                       ]))
        elif self.dataset_name == 'lsun':
            dataset = dset.LSUN(db_path=self.data_dir, classes=['bedroom_train'],
                                transform=transforms.Compose([
                                    transforms.Resize(self.imageSize),
                                    transforms.CenterCrop(self.imageSize),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]))
        elif self.dataset_name == 'cifar10':
            dataset = dset.CIFAR10(root=self.data_dir, download=True,
                                   transform=transforms.Compose([
                                       transforms.Resize(self.imageSize),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
        elif self.dataset_name == 'fake':
            dataset = dset.FakeData(image_size=(3, self.imageSize, self.imageSize),
                                    transform=transforms.ToTensor())

        elif self.dataset_name == 'humanface':
            dataset = HumanFaceDataSet(data_dir=self.data_dir,
                                       image_list_file=self.image_list_file,
                                       transform=transforms.Compose([
                                           transforms.Resize(self.imageSize),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                       ]))

        return dataset
