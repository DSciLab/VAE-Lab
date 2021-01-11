from mlutils.metrics import Accuracy
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from cfg import Opts
import mlutils
from torch.utils.data import DataLoader
from nature_datasets import get_data
from networks import get_classifier
from trainer import get_trainer
from mlutils.metrics import *


def train(opt, trainer, training_dataset, eval_dataset):
    train_loader = DataLoader(training_dataset,
                              batch_size=opt.batch_size,
                              shuffle=True,
                              num_workers=opt.num_workers,
                              pin_memory=True)

    eval_loader = DataLoader(eval_dataset,
                             batch_size=opt.batch_size,
                             shuffle=False,
                             num_workers=opt.num_workers,
                             pin_memory=True)

    trainer.train(train_loader, eval_loader)


def main(opt):
    train_dataset, eval_dataset = get_data(opt)

    trainer_cls = get_trainer(opt)
    trainer = trainer_cls(opt)

    acc_metric = Accuracy(opt)
    trainer.set_metrics(acc_metric)
    train(opt, trainer, train_dataset, eval_dataset)


if __name__ == '__main__':
    Opts.add_yaml('dataset', 'MNIST', './conf/dataset/mnist.yaml')
    Opts.add_yaml('dataset', 'CIFAR10', './conf/dataset/cifar10.yaml')
    Opts.add_yaml('dataset', 'CIFAR100', './conf/dataset/cifar100.yaml')

    Opts.add_float('lr', 0.1, 'learning rate')
    Opts.add_int('batch_size', 128, 'batch size')
    Opts.add_int('image_chan', 3, 'image channel')
    Opts.add_int('num_classes', 10, 'number of classes')
    Opts.add_int('num_workers', 5, 'number of workers')
    Opts.add_int('epochs', 100)
    Opts.add_int('device', 1)
    Opts.add_string('dataset', 'CIFAR10', 'dataset name')
    Opts.add_string('trainer', 'basic_classifier')

    opt = Opts()

    mlutils.init(opt)
    main(opt)
