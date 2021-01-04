from torchvision import datasets
import os
from torchvision import transforms


normalize = transforms.Normalize(mean=[0.1307],
                                 std=[0.30150])


def get_mnist(opt):
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

    data_root = os.path.join(opt.data_root, 'mnist')
    training_dataset = datasets.MNIST(root=data_root,
                                      train=True,
                                      transform=training_transformer,
                                      download=True)
    eval_dataset =  datasets.MNIST(root=data_root,
                                   train=False,
                                   download=True,
                                   transform=eval_transformer)


    return training_dataset, eval_dataset
