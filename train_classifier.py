from mlutils.metrics import Accuracy
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.optimizer import Optimizer
from cfg import Opts
from torch.utils.data import DataLoader
from data import get_data
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

    classifier_cls = get_classifier(opt)
    clsasifier = classifier_cls(opt)

    optimizer = SGD(clsasifier.parameters(),
                          lr=opt.lr,
                          momentum=0.9)
    scheduler = MultiStepLR(optimizer,
                            milestones=[50, 80],
                            gamma=0.1)
    
    trainer_cls = get_trainer(opt)
    trainer = trainer_cls(opt, clsasifier,
                          optimizer, scheduler)

    acc_metric = Accuracy(opt)

    trainer.add_metrics(acc_metric)
    train(opt, trainer, train_dataset, eval_dataset)


if __name__ == '__main__':
    Opts.add_float('lr', 0.1, 'learning rate')
    Opts.add_int('batch_size', 128, 'batch size')
    Opts.add_string('data_root', '/data/cwj/data/VAELab')

    opt = Opts()

    mlutils.init(opt)
    main(opt)
