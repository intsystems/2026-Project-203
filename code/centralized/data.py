from typing import Tuple
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def mnist_loaders(
    datadir: str,
    batch_size: int = 128,
    download: bool = False,
):
    """
    Централизованный MNIST, делим train/test с нормализацией
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    train_ds = datasets.MNIST(datadir, train=True, download=download, transform=transform)
    test_ds = datasets.MNIST(datadir, train=False, download=download, transform=transform)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_dl, test_dl


def cifar10_loaders(
    datadir: str,
    batch_size: int = 128,
    download: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    Централизованный CIFAR10, делим train/test с нормализацией
    """
    transform_train = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),    # Отражаем
            transforms.RandomCrop(32, padding=4), # Сдвигаем
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010),
            ),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010),
            ),
        ]
    )

    train_ds = datasets.CIFAR10(datadir, train=True, download=download, transform=transform_train)
    test_ds = datasets.CIFAR10(datadir, train=False, download=download, transform=transform_test)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_dl, test_dl

