import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np

def get_mnist_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

def get_cifar10_transforms(train: bool = True):
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


def load_full_datasets(dataset, datadir, download=False):
    if dataset == 'cifar10':
        train_ds = datasets.CIFAR10(datadir, train=True, download=download)
        test_ds = datasets.CIFAR10(datadir, train=False, download=download)
        y_train = np.array(train_ds.targets)
        y_test = np.array(test_ds.targets)
    elif dataset == 'mnist':
        train_ds = datasets.MNIST(datadir, train=True, download=download)
        test_ds = datasets.MNIST(datadir, train=False, download=download)
        y_train = train_ds.targets.numpy()
        y_test = test_ds.targets.numpy()
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
        
    return train_ds, test_ds, y_train, y_test


def get_federated_loaders(train_ds, test_ds, train_map, test_map, n_parties, batch_size):
    train_loaders = []
    test_loaders = []
    
    for i in range(n_parties):
        train_loaders.append(DataLoader(Subset(train_ds, train_map[i]), batch_size=batch_size, shuffle=True))
        test_loaders.append(DataLoader(Subset(test_ds, test_map[i]), batch_size=batch_size, shuffle=False))
    
    return train_loaders, test_loaders


def partition_data(y_train, y_test, partition, n_parties, beta=0.4):
    n_train = y_train.shape[0]
    n_test = y_test.shape[0]

    # --- Homo (IID) ---
    if partition == "homo":
        print("the homogenous dataset is being partitioned")
        idxs_train = np.random.permutation(n_train)
        idxs_test = np.random.permutation(n_test)

        batch_idxs_train = np.array_split(idxs_train, n_parties)
        batch_idxs_test = np.array_split(idxs_test, n_parties)
        
        net_dataidx_map_train = {i: batch_idxs_train[i] for i in range(n_parties)}
        net_dataidx_map_test = {i: batch_idxs_test[i] for i in range(n_parties)}

    # --- Non-IID Dirichlet ---
    elif partition == "noniid-labeldir":
        print("the non-iid label directory dataset is being partitioned")
        K = 10
        
        min_size = 0
        min_require_size = 10

        net_dataidx_map_train = {i: [] for i in range(n_parties)}
        net_dataidx_map_test = {i: [] for i in range(n_parties)}

        while min_size < min_require_size:
            idx_batch_train = [[] for _ in range(n_parties)]
            idx_batch_test = [[] for _ in range(n_parties)]
            
            for k in range(K):
                idx_k_train = np.where(y_train == k)[0]
                idx_k_test = np.where(y_test == k)[0]
                
                np.random.shuffle(idx_k_train)
                np.random.shuffle(idx_k_test)
                
                # Распределение Дирихле
                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                
                # Балансировка тренировочных данных
                proportions = np.array([p * (len(idx_j) < n_train / n_parties) for p, idx_j in zip(proportions, idx_batch_train)])
                proportions = proportions / proportions.sum()

                # Разбиваем индексы согласно пропорциям
                split_train = (np.cumsum(proportions) * len(idx_k_train)).astype(int)[:-1]
                split_test = (np.cumsum(proportions) * len(idx_k_test)).astype(int)[:-1]
                
                idx_batch_train = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch_train, np.split(idx_k_train, split_train))]
                idx_batch_test = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch_test, np.split(idx_k_test, split_test))]
            
            min_size = min([len(idx_j) for idx_j in idx_batch_train])

        for j in range(n_parties):
            net_dataidx_map_train[j] = idx_batch_train[j]
            net_dataidx_map_test[j] = idx_batch_test[j]

    return net_dataidx_map_train, net_dataidx_map_test

