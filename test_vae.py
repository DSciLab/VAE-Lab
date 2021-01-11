from torch.nn.modules import linear
import visdom
from cfg import Opts
import mlutils
from mlutils import Log
from torch.utils.data import DataLoader
from trainer import get_trainer
from nature_datasets import get_data
from networks import get_vae
import time


def inference(opt, trainer):
    _, eval_dataset = get_data(opt)
    test_loader = DataLoader(eval_dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=1,
                             pin_memory=True)

    for item in test_loader:
        Log.info('infer...')
        image, _ = item
        gen_image = trainer.infer_image(image)
        time.sleep(1)


def main(opt):
    saver = mlutils.Saver(opt)
    saver.load_cfg(opt)

    vae_cls = get_vae(opt)
    vae = vae_cls(opt)
    trainer_cls = get_trainer(opt)

    trainer = trainer_cls(opt, vae)
    trainer.load_state()

    inference(opt, trainer)


if __name__ == '__main__':
    Opts.add_int('device', 1, 'gpu id')
    Opts.add_int('dashboard_port', 10010)
    Opts.add_int('dashboard_server', False)
    Opts.add_string('normalize', 'linear')
    Opts.add_bool('test', True, 'on test stage')
    Opts.add_string('model', 'latest', 'load latest/best model.')
    Opts.add_string('id', '', required=True)

    opt = Opts()

    mlutils.init(opt)
    main(opt)
