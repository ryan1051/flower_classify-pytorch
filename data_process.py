

import numpy as np
import torch
from torchvision import datasets, transforms  # transforms包括常用的图像变换
from torch.utils.data import Subset



import json


class Dataset(object):
    def __init__(self):
        self.image_datasets = dict()
        self.dataloaders = dict()
        self.class_to_idx = dict()
        self.label_to_name = dict()

    def prepare_data(self):
        data_dir = "flower_data"

        normalize_mean = np.array([0.485, 0.456, 0.406])
        normalize_std = np.array([0.229, 0.224, 0.225])

        data_transforms = dict()

        data_transforms['train'] = transforms.Compose([  # 多个transform组合起来使用
            transforms.RandomChoice([  # 从给定的方式中选一个操作
                transforms.RandomHorizontalFlip(p=0.5),  # 按概率翻转
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(180),
            ]),
            transforms.RandomResizedCrop(224),  # 先随机裁剪再resize到指定大小
            transforms.ToTensor(),  # 转为tensor并归一化至【0-1】
            transforms.Normalize(  # 归一化，先减均值再除以标准差
                normalize_mean,
                normalize_std)
            ])

        data_transforms['valid'] = transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),  # 中心裁剪
            transforms.ToTensor(),
            transforms.Normalize(
                normalize_mean,
                normalize_std)
            ])

        self.image_datasets['train_data'] = datasets.ImageFolder(data_dir + '/valid',
                                                                 transform=data_transforms['train'])
        valid_dataset_to_split = datasets.ImageFolder(data_dir + '/valid',
                                                      transform=data_transforms['valid'])

        valid_data_index_list = []
        test_data_index_list = []
        for index in range(0, len(valid_dataset_to_split), 2):
            valid_data_index_list.append(index)
            test_data_index_list.append(index+1)

        self.image_datasets['valid_data'] = Subset(valid_dataset_to_split, valid_data_index_list)
        self.image_datasets['test_data'] = Subset(valid_dataset_to_split, test_data_index_list)

        self.dataloaders['train_data'] = torch.utils.data.DataLoader(self.image_datasets['train_data'], batch_size=16,
                                                                     shuffle=True, num_workers=32)
        self.dataloaders['valid_data'] = torch.utils.data.DataLoader(self.image_datasets['valid_data'], batch_size=16,
                                                                     shuffle=False, num_workers=32)
        self.dataloaders['test_data'] = torch.utils.data.DataLoader(self.image_datasets['test_data'], batch_size=16,
                                                                    shuffle=False, num_workers=32)

        print("Train data: {} images / {} batches".format(len(self.dataloaders['train_data'].dataset),
                                                          len(self.dataloaders['train_data'])))
        print("Valid data: {} images / {} batches".format(len(self.dataloaders['valid_data'].dataset),
                                                          len(self.dataloaders['valid_data'])))
        print("Test  data: {} images / {} batches".format(len(self.dataloaders['test_data'].dataset),
                                                          len(self.dataloaders['test_data'])))

    def get_label2name(self):
        with open('id2label.json', 'r') as f:
            id2label = json.load(f)

        self.class_to_idx = self.image_datasets['train_data'].class_to_idx

        for label, indx in self.class_to_idx.items():
            name = id2label.get(label)
            self.label_to_name[indx] = name

        print(self.label_to_name)

