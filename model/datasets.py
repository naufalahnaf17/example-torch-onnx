import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

def get_datasets():
    train_data = datasets.MNIST(
        root="data",
        download=True,
        train=True,
        transform=ToTensor()
    )

    test_data = datasets.MNIST(
        root="data",
        download=True,
        train=False,
        transform=ToTensor()
    )

    return train_data,test_data

def transform_to_dataLoader(data,batch_size,shuffle):
    torch.manual_seed(42)

    loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle
    )

    return loader