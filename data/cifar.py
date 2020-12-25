from torchvision import datasets
import os
from torch.utils.data import DataLoader
from torchvision import transforms


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def get_cifar10(opt):
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

    data_root = os.path.join(opt.data_root, 'cifar10')
    training_dataset = datasets.CIFAR10(root=opt.data_root,
                                        train=True,
                                        transform=training_transformer,
                                        download=True)
    eval_dataset =  datasets.CIFAR10(root=opt.data_root, 
                                     train=False,
                                     download=True,
                                     transform=eval_transformer),

    train_loader = DataLoader(training_dataset,
                              batch_size=opt.batch_size,
                              shuffle=True,
                              num_workers=opt.num_workers,
                              pin_memory=True)

    eval_loader = DataLoader(eval_dataset,
                             batch_size=128,
                             shuffle=False,
                             num_workers=opt.num_workers,
                             pin_memory=True)

    return train_loader, eval_loader



def get_cifar100(opt):
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
                                      transform=eval_transformer),

    train_loader = DataLoader(training_dataset,
                              batch_size=opt.batch_size,
                              shuffle=True,
                              num_workers=opt.num_workers,
                              pin_memory=True)

    eval_loader = DataLoader(eval_dataset,
                             batch_size=128,
                             shuffle=False,
                             num_workers=opt.num_workers,
                             pin_memory=True)

    return train_loader, eval_loader
