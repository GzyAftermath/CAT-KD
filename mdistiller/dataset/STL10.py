import os
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import torch

data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/STL-10')

def get_STL10_dataloaders(batch_size, val_batch_size, num_workers):    
    train_set = datasets.STL10(root=data_folder,
                              split='train',
                              transform=transforms.Compose([
                                  transforms.Resize((32,32)),
                                  transforms.RandomCrop(32, padding=4),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                              )
    num_data = len(train_set)
    test_set = datasets.STL10(root=data_folder,
                              split='test',
                              transform=transforms.Compose([
                                  transforms.Resize((32,32)),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=1,
    )
    return train_loader, test_loader, num_data