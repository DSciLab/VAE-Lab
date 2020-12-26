from .classifier_trainer import ClassificationTrainer
from .vae_trainer import VAETrainer


def get_trainer(opt):
    if opt.trainer == 'classification':
        return ClassificationTrainer
    elif opt.trainer == 'vae':
        return VAETrainer
    else:
        raise RuntimeError(f'Unrecognized trainer [{opt.trainer}]')
