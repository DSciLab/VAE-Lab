from torch.nn import functional as F
from torch.optim import Adam
from mlutils import Trainer
from networks.vae import vanilla_d_c
from .losses import KLDLoss


class VAECTrainer(Trainer):
    def __init__(self, opt):
        super().__init__(opt)
        self.kld_weight = opt.kld_weight
        vae = vanilla_d_c.VanillaVAE(opt)
        clf = vanilla_d_c.Classifier(opt)
        self.vae = self.to_gpu(vae)
        self.clf = self.to_gpu(clf)
        self.optimizer = Adam(list(vae.parameters()) + list(clf.parameters()),
                              lr=opt.lr)
        # self.scheduler = MultiStepLR(self.optimizer,
        #                              milestones=[50, 80],
        #                              gamma=0.1)
        self.kld_loss = KLDLoss(opt)
        self.bce_loss = F.binary_cross_entropy
        self.ce_loss = F.cross_entropy
        self.mse_loss = F.mse_loss

    def train_step(self, item):
        images, labels = item
        images = self.to_gpu(images)
        labels = self.to_gpu(labels)

        self.optimizer.zero_grad()
        mu, logvar, fake, z = self.vae(images)
        # train generator
        kld_loss = self.kld_loss(mu, logvar)
        recons_loss = self.mse_loss(fake, images)
        vae_loss = kld_loss * self.kld_weight + recons_loss

        # train classifier
        pred = self.clf(z)
        clf_loss = self.ce_loss(pred, labels)
        loss = vae_loss + clf_loss
        loss.backward()
        self.optimizer.step()

        self.dashboard.add_trace_dict({'recons_loss': recons_loss.detach(),
                                       'kld_loss': kld_loss.detach(),
                                       'clf_loss': clf_loss.detach()}, 
                                      self.step)
        return loss.detach(), None, None

    def eval_step(self, item):
        images, labels = item
        images = self.to_gpu(images)
        labels = self.to_gpu(labels)

        mu, logvar, fake, z = self.vae(images)
        # eval generator
        kld_loss = self.kld_loss(mu, logvar)
        recons_loss = self.mse_loss(fake, images)
        vae_loss = kld_loss * self.kld_weight + recons_loss

        # train classifier
        pred = self.clf(z)
        clf_loss = self.ce_loss(pred, labels)
        loss = vae_loss + clf_loss

        self.dashboard.add_image_dict({'eval_fake': fake,
                                       'eval_image': images})

        return loss.detach(), None, None

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
