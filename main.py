import torch
import torch.nn as nn
import torch.utils
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import cv2
import time
import os
import copy
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import random


# # office pc
# DATA_DIR = '/home/user/Datasets'  
# RUNS_DIR = '/home/user/Projects/runs/pytorch-ex2/exp3'
# laptop:
DATA_DIR = '/home/roni/Projects/Data'  
RUNS_DIR = '/home/roni/Projects/runs/pytorch-ex2'



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_data_loaders(args:dict) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]: 
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8,1.0), ratio=(0.8,1.2)),
            # transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(5),
            transforms.ToTensor()
        ]), 

        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
    }

    train_dataset_temp = datasets.MNIST(root=args.data_path, download=False, train=True, transform=data_transforms['train'])
    val_dataset_temp = datasets.MNIST(root=args.data_path, download=False, train=True, transform=data_transforms['test'])
    val_size = len(train_dataset_temp)//5
    indices = torch.randperm(len(train_dataset_temp))
    train_dataset = Subset(train_dataset_temp, indices=indices[:-val_size])
    val_dataset = Subset(val_dataset_temp, indices=indices[-val_size:])

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    test_dataset = datasets.MNIST(root=args.data_path, download=False, train=False, transform=data_transforms['test'])
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    dataloaders = {
        'train': train_dataloader, 
        'val':   val_dataloader, 
        'test':  test_dataloader 
    }

    class_names = train_dataset_temp.classes

    return dataloaders, class_names


def train_loop(model:nn.Module, criterion:nn.Module, dataloaders:dict[DataLoader], optimizer:torch.optim.Optimizer, 
               num_epochs:int, start_epoch:int, save_dir:str, scheduler=None):

    global tb_writer

    best_model_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    layout = {
        "training": {
            "loss": ["Multiline", ["loss/train", "loss/val"]],
            "accuracy": ["Multiline", ["accuracy/train", "accuracy/val"]],
        },
    }
    tb_writer.add_custom_scalars(layout)

    training_start_time = time.time()

    for epoch in range(start_epoch, num_epochs):
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
            n_total_steps = len(dataloaders[phase])

            #iterate over the data:
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
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
                iter_loss = loss.item()
                iter_accuracy = torch.sum(pred==labels.data).double() / len(labels)

                tb_writer.add_scalar(f'loss/{phase}', iter_loss, epoch * n_total_steps + i)
                tb_writer.add_scalar(f'accuracy/{phase}', iter_accuracy, epoch * n_total_steps + i)
                
            if phase=='train' and scheduler:
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_correct.double() / len(dataloaders[phase].dataset)
            print(f'{phase} Loss: {epoch_loss:.4f} ,  Acc: {epoch_acc:.4f} ,  Phase-time: {(time.time()-phase_start_time):.2f} sec, Dataset-Size: {len(dataloaders[phase].dataset)}')
            if phase=='train':
                epoch_train_loss = epoch_loss
                epoch_train_acc = epoch_acc
            else:
                epoch_val_loss = epoch_loss
                epoch_val_acc = epoch_acc

            if phase=='val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())
                torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': epoch_val_loss,
                            'accurcay': epoch_val_acc
                            }, os.path.join(save_dir, 'best_model.pt'))

        tb_writer.add_scalars('epoch-loss', {'train':epoch_train_loss, 'val':epoch_val_loss}, epoch)
        tb_writer.add_scalars('epoch-accuracy', {'train':epoch_train_acc, 'val':epoch_val_acc}, epoch)
        tb_writer.flush()

        print()

    training_time = time.time() - training_start_time
    print(f'Training completed in {int(training_time//60)} minutes and {int(training_time%60)} seconds')
        
    #load the weights of the best model:
    model.load_state_dict(best_model_weights)
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_val_loss,
                'accurcay': epoch_val_acc
                }, os.path.join(save_dir, 'final_model.pt'))

    return model



def test_loop(model:nn.Module, dataloader:DataLoader):

    global tb_writer

    start_time = time.time()
    model.eval()

    running_correct = 0
    n_total_steps = len(dataloader)

    #iterate over the data:
    for i, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        #forward:
        #calc gradients only if in training phase
        with torch.no_grad():
            outputs = model(inputs)
            _, pred = torch.max(outputs, 1)
            
            
        #statistics:
        running_correct += torch.sum(pred==labels.data)
        
    accuracy = running_correct.double() / len(dataloader.dataset)
    print(f'Test Set Accuracy: {accuracy:.4f} ,  Phase-time: {(time.time()-start_time):.2f} sec, Dataset-Size: {len(dataloader.dataset)}')

    return accuracy


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


def print_grid_images(example_data):
    
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.imshow(example_data[i][0], cmap='gray')
    plt.show()


def increment_name(path):
    '''increase save directory's id'''
    path = Path(path)
    sep = ''
    if path.exists():
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        for n in range(1, 9999):
            p = f'{path}{sep}{n}{suffix}'
            if not os.path.exists(p):
                break
        path = Path(p)
    return path


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='Pytorch-ex2 Training', add_help=add_help)
    parser.add_argument('--data-path', default=DATA_DIR, type=str, help='path of dataset')
    parser.add_argument('--output-dir', default=RUNS_DIR, type=str, help='path to save outputs')
    parser.add_argument('--name', default='exp', type=str, help='experiment name, saved to output_dir/name')
    parser.add_argument('--device', default='0', type=str, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--batch-size', default=32, type=int, help='total batch size for all GPUs')
    parser.add_argument('--workers', default=8, type=int, help='number of data loading workers (default: 8)')
    parser.add_argument('--epochs', default=1000, type=int, help='number of total epochs to run')
    parser.add_argument('--learning-rate', default=0.01, type=float, help='optimizer learning rate (default: 0.01)')
    parser.add_argument('--checkpoint-file', default='', type=str, help='path to .pt file with model and optimizer params to be loaded')
    parser.add_argument('--test', nargs='?', const=True, default=False, help='run on test data')

    # parser.add_argument('--eval-interval', default=20, type=int, help='evaluate at every interval epochs')
    # parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume the most recent training')
    # parser.add_argument('--run_debugpy', nargs='?', const=True, default=False, help='run debugpy service to enable debugging through VS-Code')
    
    return parser


def main(args, seed:int=0):

    # Fixed random number seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    global device
    device = torch.device('cuda:'+args.device if torch.cuda.is_available() and args.device.isdigit() else 'cpu')

    if not args.test: 
        save_dir = str(increment_name(os.path.join(args.output_dir, args.name)))
        os.mkdir(save_dir)
        
        tb_path = os.path.join(save_dir, 'tensorboard')
        os.mkdir(tb_path)
        global tb_writer
        tb_writer = SummaryWriter(tb_path)

    dataloaders, class_names = get_data_loaders(args)
    print(f'classes = {class_names}')
    example_data, _ = next(iter(dataloaders['test']))
    image_grid = torchvision.utils.make_grid(example_data)
    # tb_writer.add_image('example images', image_grid)
    
    checkpoint = []
    model = MyConvNet()
    if len(args.checkpoint_file):
        checkpoint = torch.load(args.checkpoint_file)
        model.load_state_dict(checkpoint['model_state_dict'])

    model = model.to(device)

    if args.test:
        test_loop(model, dataloaders['test'])
        return
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    start_epoch = 0
    if checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        optimizer.param_groups[0]['lr'] = args.learning_rate


    tb_writer.add_graph(model, example_data.to(device))
    # tb_writer.flush()

    train_loop(model, dataloaders=dataloaders, criterion=criterion, optimizer=optimizer, 
               num_epochs=args.epochs, start_epoch=start_epoch, 
               save_dir=save_dir)

    tb_writer.close()



if __name__ == '__main__':

    print(f'device = {device}')

    args = get_args_parser().parse_args()
    main(args)
    print('done.')

