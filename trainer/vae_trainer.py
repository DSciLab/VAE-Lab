import torch
from torch.nn import functional as F
from mlutils import Trainer
from mlutils import Log


def vae_ll_loss(gen_images, images, mu, logvar,
                kld_weight=1):
    recons_loss = F.mse_loss(gen_images, images)
    kld_loss = torch.mean(-0.5 * \
        torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim = 1), dim = 0)

    Log.debug(images)
    Log.debug(gen_images)
    Log.debug(recons_loss)
    Log.debug(kld_loss)

    loss = recons_loss + kld_loss * kld_weight
    return loss, recons_loss, kld_loss


class VAETrainer(Trainer):
    def __init__(self, opt, model, optimizer=None, scheduler=None):
        super().__init__(opt)
        self.kld_weight = opt.kld_weight
        self.model = self.to_gpu(model)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = vae_ll_loss

    def train_step(self, item):
        images, labels = item
        images = self.to_gpu(images)
        labels = self.to_gpu(labels)

        self.optimizer.zero_grad()
        mu, logvar, gen_images = self.model(images)

        loss, recons_loss, kld_loss = self.loss_fn(gen_images, images, mu, logvar, self.kld_weight)
        loss.backward()
        self.optimizer.step()

        self.dashboard.add_trace_dict({'recons_loss': recons_loss.detach(),
                                        'kld_loss': kld_loss.detach()}, self.step)
        # self.dashboard.add_image_dict({'train_gen_image': gen_images,
        #                                 'train_image': images})
        return loss.detach(), gen_images, images

    def eval_step(self, item):
        images, labels = item
        images = self.to_gpu(images)
        labels = self.to_gpu(labels)

        mu, logvar, gen_images = self.model(images)
        loss, recons_loss, kld_loss = self.loss_fn(gen_images, images, mu, logvar, self.kld_weight)

        self.dashboard.add_image_dict({'eval_gen_image': gen_images,
                                       'eval_image': images})
        return loss.detach(), gen_images, images

    def infer(self, item):
        images, labels = item
        images = self.to_gpu(images)
        labels = self.to_gpu(labels)

        mu, logvar, gen_images = self.model(images)
        loss = self.loss_fn(gen_images, images, mu, logvar)

        # self.dashboard.add_image_dict({'gen_image': gen_images,
        #                                 'image': images})
        return loss.detach(), gen_images, images

    def infer_image(self, image):
        image = self.to_gpu(image)
        _, _, gen_image = self.model(image)
        self.dashboard.add_image_dict(
                            {'gen_image': gen_image,
                             'image': image})
        return gen_image

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self):
        pass

    def on_training_begin(self):
        pass

    def on_training_end(self):
        pass
