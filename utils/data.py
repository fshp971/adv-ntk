from PIL import Image
import numpy as np
import torch
import torchvision
import os
import pickle


class Dataset():
    def __init__(self, x, y, transform=None):
        self.img_mode = None
        if x.shape[-1] == 1:
            self.img_mode = "L"
            x = np.squeeze(x, axis=-1)

        self.x = x
        self.y = y
        self.transform = transform

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]
        if self.transform is not None:
            x = self.transform( Image.fromarray(x, mode=self.img_mode) )
        return x, y

    def __len__(self):
        return len(self.x)


class IndexedDataset():
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return x, y, idx

    def __len__(self):
        return len(self.dataset)


class SubDataset:
    def __init__(self, dataset, size):
        self.dataset = dataset
        self.indices = np.random.permutation(np.arange(len(dataset)))[:size]

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


def datasetCIFAR10(root='./data', train=True, transform=None):
    return torchvision.datasets.CIFAR10(root=root, train=train,
                        transform=transform, download=True)


def datasetCIFAR100(root='./data', train=True, transform=None):
    return torchvision.datasets.CIFAR100(root=root, train=train,
                        transform=transform, download=True)


def datasetMNIST(root='./data', train=True, transform=None):
    return torchvision.datasets.MNIST(root=root, train=train,
                        transform=transform, download=True)


def datasetFashionMNIST(root='./data', train=True, transform=None):
    return torchvision.datasets.FashionMNIST(root=root, train=train,
                        transform=transform, download=True)


def datasetSVHN(root='./path', train=True, transform=None):
    return torchvision.datasets.SVHN(root=root, split='train' if train else 'test',
                        transform=transform, download=True)


class Loader():
    def __init__(self, dataset, batch_size, shuffle=False, drop_last=False, num_workers=4):
        self.loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
        self.iterator = None

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)

    def __next__(self):
        if self.iterator is None:
            self.iterator = iter(self.loader)

        try:
            samples = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            samples = next(self.iterator)

        return samples
