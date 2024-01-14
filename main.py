import torch
import torch.nn as nn
import torch.utils
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader, Subset

import numpy as np
import cv2
import time
import os


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_data_loaders(data_dir:str) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]: 
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]), 

        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
    }

    train_dataset_temp = datasets.MNIST(root=data_dir, download=False, train=True, transform=data_transforms['train'])
    val_dataset_temp = datasets.MNIST(root=data_dir, download=False, train=True, transform=data_transforms['test'])
    val_size = len(train_dataset_temp)//5
    indices = torch.randperm(len(train_dataset_temp))
    train_dataset = Subset(train_dataset_temp, indices=indices[:-val_size])
    val_dataset = Subset(val_dataset_temp, indices=indices[-val_size:])

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

    test_dataset = datasets.MNIST(root=data_dir, download=False, train=False, transform=data_transforms['test'])
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    class_names = train_dataset_temp.classes

    return train_dataloader, val_dataloader, test_dataloader, class_names



if __name__ == '__main__':

    print(f'device = {device}')

    data_dir = '/home/user/Datasets'
    train_dataloader, val_dataloader, test_dataloader, class_names = get_data_loaders(data_dir)
    print(f'classes = {class_names}')


    print('done.')