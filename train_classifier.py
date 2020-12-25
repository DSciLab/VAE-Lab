import torch
from cfg import Opts
from .data import get_data
from .networks import get_classifier


def train(trainer, train_dataloader, eval_dataloader):
    pass


def main(opt):
    classifier_cls = get_classifier(opt)
    clsasifier = classifier_cls(opt)
    train_loader, eval_loader = get_data(opt)
    train()


if __name__ == '__main__':
    pass
