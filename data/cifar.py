import os
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms.transforms import Resize

from .utils import LinearNormalize


normalize_fn = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
linear_normalize_fn = LinearNormalize()


def get_cifar10(opt):
    if opt.get('normalize', 'normalize') == 'linear':
        normalize = linear_normalize_fn
    else:
        normalize = normalize_fn

    training_transformer = transforms.Compose([
                            transforms.RandomHorizontalFlip(),
                            transforms.Resize(64),
                            transforms.RandomCrop(64, 4),
                            transforms.RandomRotation(15),
                            transforms.ToTensor(),
                            normalize,
                        ])
    eval_transformer = transforms.Compose([
                        transforms.Resize(64),
                        transforms.ToTensor(),
                        normalize,
                    ])

    data_root = os.path.join(opt.data_root, 'cifar10')
    training_dataset = datasets.CIFAR10(root=data_root,
                                        train=True,
                                        transform=training_transformer,
                                        download=True)
    eval_dataset =  datasets.CIFAR10(root=data_root, 
                                     train=False,
                                     download=True,
                                     transform=eval_transformer)

    return training_dataset, eval_dataset



def get_cifar100(opt):
    if opt.get('normalize', 'normalize') == 'linear':
        normalize = linear_normalize_fn
    else:
        normalize = normalize_fn

    training_transformer = transforms.Compose([
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomCrop(32, 4),
                            transforms.RandomRotation(15),
                            transforms.ToTensor(),
                            normalize,
                        ])
    eval_transformer = transforms.Compose([
                        transforms.ToTensor(),
                        normalize,
                    ])

    data_root = os.path.join(opt.data_root, 'cifar100')
    training_dataset = datasets.CIFAR100(root=data_root,
                                         train=True,
                                         transform=training_transformer,
                                         download=True)
    eval_dataset =  datasets.CIFAR100(root=data_root, 
                                      train=False,
                                      download=True,
                                      transform=eval_transformer)

    return training_dataset, eval_dataset
