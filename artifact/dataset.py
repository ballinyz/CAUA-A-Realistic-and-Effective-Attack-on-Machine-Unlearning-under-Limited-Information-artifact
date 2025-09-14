
import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from torchvision import datasets, transforms
import torch
from torch.utils.data import Subset
import os

# 数据集路径（根据需要修改）
dataset_path = 'D:/pythonp/alldataset'

# 数据集注册表和装饰器
_dataset_registry = {}
def _add_dataset(name):
    def decorator(fn):
        _dataset_registry[name.lower()] = fn
        return fn
    return decorator

# --------------------------- 各数据集加载函数 ---------------------------

@_add_dataset('cifar10')
def load_cifar10_dataset(transform, seed=42):
    train_set = datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root=dataset_path, train=False, download=True, transform=transform)
    return train_set, test_set

@_add_dataset('cifar100')
def load_cifar100_dataset(transform, seed=42):
    train_set = datasets.CIFAR100(root=dataset_path, train=True, download=True, transform=transform)
    test_set = datasets.CIFAR100(root=dataset_path, train=False, download=True, transform=transform)
    return train_set, test_set

@_add_dataset('stl10')
def load_stl10_dataset(transform, seed=42):
    train_set = datasets.STL10(root=dataset_path, split='train', download=True, transform=transform)
    test_set = datasets.STL10(root=dataset_path, split='test', download=True, transform=transform)
    return train_set, test_set

# --------------------------- 通用 transform 定义 ---------------------------

def get_transform(name):
    if name == 'stl10':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))
        ])
    elif name == 'cifar100':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                                 (0.2675, 0.2565, 0.2761))
        ])
    else:  # 默认处理，如 cifar10
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))
        ])

# --------------------------- Tensor 转换 ---------------------------

def to_tensor(dataset):
    x = torch.stack([item[0] for item in dataset])
    y = torch.tensor([item[1] for item in dataset])
    return x, y

# --------------------------- 主调度函数 ---------------------------
def load_dataset(name='cifar10', seed=42):
    name = name.lower()
    if name not in _dataset_registry:
        raise ValueError(f"Dataset {name} is not registered.")

    transform = get_transform(name)
    loader_fn = _dataset_registry[name]
    train_set, test_set = loader_fn(transform, seed)

    x_train, y_train = to_tensor(train_set)
    x_test, y_test = to_tensor(test_set) if test_set else (None, None)

    return (x_train, y_train), (x_test, y_test)





