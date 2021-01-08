from mlutils.metrics import Accuracy
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR
from cfg import Opts
import mlutils
from mlutils import Log
from torch.utils.data import DataLoader
from data import get_data
from networks import get_vae
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

    vae_cls = get_vae(opt)
    vae = vae_cls(opt)

    optimizer = Adam(vae.parameters(),
                          lr=opt.lr)
    scheduler = MultiStepLR(optimizer,
                            milestones=[50, 80],
                            gamma=0.1)
    
    trainer_cls = get_trainer(opt)
    trainer = trainer_cls(opt, vae,
                          optimizer, scheduler)

    # trainer.set_metrics()
    train(opt, trainer, train_dataset, eval_dataset)


if __name__ == '__main__':
    Opts.add_float('lr', 0.005, 'learning rate')
    Opts.add_int('batch_size', 256, 'batch size')
    Opts.add_int('image_chan', 3, 'image channel')
    Opts.add_int('num_classes', 10, 'number of classes')
    Opts.add_int('num_workers', 5, 'number of workers')
    Opts.add_int('epochs', 10000)
    Opts.add_int('device', 1)
    Opts.add_int('width', 64, 'image width')
    Opts.add_int('z_dim', 128, 'latent space dim')
    Opts.add_int('kld_weight', 0.15, 'KLD Loss weight')
    Opts.add_bool('debug', False)
    Opts.add_bool('dashboard', True, 'enable/disable dashboard.')
    Opts.add_int('dashboard_port', 10010)
    Opts.add_int('dashboard_server', False)
    Opts.add_string('normalize', 'linear')
    Opts.add_string('dataset', 'CELEBA', 'dataset name')
    Opts.add_string('data_root', '/data/cwj/data/VAELab')
    Opts.add_string('trainer', 'vae')
    Opts.add_string('vae', 'vanilla')

    opt = Opts()

    if opt.debug:
        Log.set_level(Log.DEBUG)

    mlutils.init(opt)
    main(opt)
