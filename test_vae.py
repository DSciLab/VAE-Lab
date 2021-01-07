from numpy.lib.type_check import imag
from cfg import Opts
import mlutils
from torch.utils.data import DataLoader
from trainer import get_trainer
from data import get_data
import time


def inference(opt, trainer):
    _, eval_dataset = get_data(opt)
    test_loader = DataLoader(eval_dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=1,
                             pin_memory=True)

    for item in test_loader:
        image, _ = item
        gen_image = trainer.infer_image(image)
        time.sleep(1)


def main(opt):
    saver = mlutils.Saver(opt)
    saver.load_cfg(opt)
    trainer_cls = get_trainer(opt)
    trainer = trainer_cls(opt)
    trainer.load_state()

    inference(opt, trainer)


if __name__ == '__main__':
    Opts.add_int('device', 1, 'gpu id')
    Opts.add_bool('test', True, 'on test stage')
    Opts.add_string('model', 'latest', 'load latest/best model.')
    Opts.add_string('id', '', required=True)

    opt = Opts()

    mlutils.init(opt)
    main(opt)
