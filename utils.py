# some useful functions

import math
import torch
from torchvision import datasets, transforms
import numpy as np


def accuracy(out, y):
    _, pred = out.max(1)
    correct = pred.eq(y)
    return 100 * correct.sum().float() / y.size(0)

def get_data(args):

    # mean/std stats
    if args.dataset == 'cifar10':
        data_class = 'CIFAR10'
        num_classes = 10
        input_dim = 32 * 32 * 3
        stats = {
            'mean': [0.491, 0.482, 0.447],
            'std': [0.247, 0.243, 0.262]
            }
    elif args.dataset == 'cifar100':
        data_class = 'CIFAR100'
        num_classes = 100
        input_dim = 32 * 32 * 3
        stats = {
            'mean': [0.5071, 0.4867, 0.4408] ,
            'std': [0.2675, 0.2565, 0.2761]
            }
    elif args.dataset == 'mnist':
        data_class = 'MNIST'
        num_classes = 10
        input_dim = 28 * 28
        stats = {
            'mean': [0.1307],
            'std': [0.3081]
            }
    else:
        raise ValueError("unknown dataset")

    # input transformation w/o preprocessing for now

    trans = [
        transforms.ToTensor(),
        lambda t: t.type(torch.get_default_dtype()),
        transforms.Normalize(**stats)
        ]

    # get tr and te data with the same normalization
    # no preprocessing for now
    tr_data = getattr(datasets, data_class)(
        root=args.path,
        train=True,
        download=True,
        transform=transforms.Compose(trans)
        )

    te_data = getattr(datasets, data_class)(
        root=args.path,
        train=False,
        download=True,
        transform=transforms.Compose(trans)
        )

    # get tr_loader for train/eval and te_loader for eval
    train_loader = torch.utils.data.DataLoader(
        dataset=tr_data,
        batch_size=args.batch_size_train,
        shuffle=True,
        )

    train_loader_eval = torch.utils.data.DataLoader(
        dataset=tr_data,
        batch_size=args.batch_size_eval,
        shuffle=True,
        )

    test_loader_eval = torch.utils.data.DataLoader(
        dataset=te_data,
        batch_size=args.batch_size_eval,
        shuffle=True,
        )
    return train_loader, test_loader_eval, train_loader_eval, num_classes, input_dim
