from torchvision import datasets
import os
from torch.utils.data import DataLoader
from torchvision import transforms


normalize = transforms.Normalize(mean=[0.1307],
                                 std=[0.30150])


def get_fashion_mnist(opt):
    training_transformer = transforms.Compose([
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomCrop(28, 4),
                            transforms.RandomRotation(15),
                            transforms.ToTensor(),
                            normalize,
                        ])
    eval_transformer = transforms.Compose([
                        transforms.ToTensor(),
                        normalize,
                    ])

    data_root = os.path.join(opt.data_root, 'fashion_mnist')
    training_dataset = datasets.FashionMNIST(root=data_root,
                                             train=True,
                                             transform=training_transformer,
                                             download=True)
    eval_dataset =  datasets.FashionMNIST(root=data_root, 
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
