from utils.trainer import VAETrainer
from cfg import Opts
from models.basic_vae import BasicVAE
from utils.trainer import VAETrainer


def train(trainer):
    pass


def main(opt):
    vae = BasicVAE(opt)
    trainer = VAETrainer(opt)



if __name__ == '__main__':
    Opts.add_float('lr', 'leaning rate', 0.1)

    opt = Opts()
    main(opt)
