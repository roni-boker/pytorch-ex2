import torch
import torch.nn as nn
import torch.utils
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn.functional as F

import numpy as np
import cv2
import time
import os
import copy


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

    dataloaders = {
        'train': train_dataloader, 
        'val':   val_dataloader, 
        'test':  test_dataloader 
    }

    class_names = train_dataset_temp.classes

    return dataloaders, class_names


def train_loop(model:nn.Module, criterion:nn.Module, dataloaders:dict[DataLoader], optimizer:torch.optim.Optimizer, scheduler=None, num_epochs:int=3):

    best_model_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    training_start_time = time.time()

    for epoch in range(num_epochs):
        print(f'epoch {epoch} / {num_epochs-1}')
        print('-' * 10)

        #each epoch has a training phase followed by a validation phase
        for phase in ['train', 'val']:
            phase_start_time = time.time()

            if phase=='train':
                model.train()
            else:
                model.eval()
        
            running_loss = 0.0
            running_correct = 0

            #iterate over the data:
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                #forward:
                #calc gradients only if in training phase
                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(inputs)
                    _, pred = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase=='train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                #statistics:
                running_loss += loss.item() * inputs.size(0)
                running_correct += torch.sum(pred==labels.data)

            if phase=='train' and scheduler:
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_correct.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} ,  Acc: {epoch_acc:.4f} ,  Phase-time: {(time.time()-phase_start_time):.2f} sec, Dataset-Size: {len(dataloaders[phase].dataset)}')

            if phase=='val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())

        print()

    training_time = time.time() - training_start_time
    print(f'Training completed in {int(training_time//60)} minutes and {int(training_time%60)} seconds')
        
    #load the weights of the best model:
    model.load_state_dict(best_model_weights)

    return model


class MyConvNet(nn.Module):

    def __init__(self):
        super(MyConvNet, self).__init__()
        
        input_channels = 1
        hidden_layers_num = 4
        channels_num = 20
        classes_num = 10

        self.first_layer = nn.Conv2d(input_channels, channels_num, kernel_size=3, padding=1)
        self.max_pool = nn.MaxPool2d(2, 2)
        conv_layers = []
        for i in range(hidden_layers_num):
            conv_layers.append(nn.Conv2d(channels_num, channels_num, kernel_size=3, padding=1))
        self.conv_layers = nn.ModuleList(conv_layers)
        self.flatten_layer = nn.Flatten()
        d = int(max(1, np.floor(224 / 2**(1+hidden_layers_num))))
        self.last_layer = nn.Linear(channels_num*d*d, classes_num)


    def forward(self, x):

        x = self.max_pool(F.relu(self.first_layer(x)))

        for layer in self.conv_layers:
            x = F.relu(layer(x))
            if x.shape[-1] >= 2:
                x = self.max_pool(x)

        x = self.flatten_layer(x)
        x = self.last_layer(x)

        return x



if __name__ == '__main__':

    print(f'device = {device}')

    data_dir = '/home/user/Datasets'
    learning_rate = 0.01

    dataloaders, class_names = get_data_loaders(data_dir)
    print(f'classes = {class_names}')

    model = MyConvNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    train_loop(model, dataloaders=dataloaders, criterion=criterion, optimizer=optimizer)

    print('done.')