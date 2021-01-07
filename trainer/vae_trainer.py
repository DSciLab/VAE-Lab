import torch
from torch import nn
from mlutils import Trainer


def vae_ll_loss(gen_images, images, mu, logvar):
    batch_size = images.shape[0]
    sse_loss = nn.MSELoss(reduction = 'sum') # sum of squared errors
    KLD = 1. / batch_size * -0.5 * \
        torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    mse = 1. / batch_size * sse_loss(gen_images, images)
    auto_loss = mse + KLD
    return auto_loss, mse, KLD


class VAETrainer(Trainer):
    def __init__(self, opt, model, optimizer, scheduler):
        super().__init__(opt)
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
        loss, mse, kld = self.loss_fn(gen_images, images, mu, logvar)
        loss.backward()
        self.optimizer.step()

        # self.dashboard.add_image_dict({'gen_image': gen_images,
        #                                 'image': images})
        return loss.detach(), gen_images, images

    def eval_step(self, item):
        images, labels = item
        images = self.to_gpu(images)
        labels = self.to_gpu(labels)

        mu, logvar, gen_images = self.model(images)
        loss = self.loss_fn(gen_images, images, mu, logvar)

        # self.dashboard.add_image_dict({'gen_image': gen_images,
        #                                 'image': images})
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
