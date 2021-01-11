import torch
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch import autograd
from mlutils import Trainer
from networks.vae import vanilla_d_c
from .losses import KLDLoss


class VAEDTrainer(Trainer):
    def __init__(self, opt):
        super().__init__(opt)
        self.kld_weight = opt.kld_weight
        vae = vanilla_d_c.VanillaVAE(opt)
        disc = vanilla_d_c.Discriminator(opt)
        clf = vanilla_d_c.Classifier(opt)
        self.vae = self.to_gpu(vae)
        self.disc = self.to_gpu(disc)
        self.clf = self.to_gpu(clf)
        self.optimizerG = Adam(list(vae.parameters()) + \
                               list(clf.parameters()),
                               lr=opt.lr)
        self.optimizerD = Adam(disc.parameters(), lr=opt.lr)
        # self.scheduler = MultiStepLR(self.optimizer,
        #                              milestones=[50, 80],
        #                              gamma=0.1)
        self.kld_loss = KLDLoss(opt)
        self.bce_loss = F.binary_cross_entropy
        self.ce_loss = F.cross_entropy

    def update_label(self, batch_size):
        self.valid = autograd.Variable(
                        torch.Tensor(batch_size, 1).fill_(1.0), 
                        requires_grad=False)
        self.fake = autograd.Variable(
                        torch.Tensor(batch_size, 1).fill_(0.0),
                        requires_grad=False)
        
        self.valid = self.to_gpu(self.valid)
        self.fake = self.to_gpu(self.fake)

    def train_step(self, item):
        images, labels = item
        images = self.to_gpu(images)
        labels = self.to_gpu(labels)
        batch_size = images.size(0)
        self.update_label(batch_size)


        mu, logvar, fake, z = self.vae(images)
        # train generator
        self.optimizerG.zero_grad()
        kld_loss = self.kld_loss(mu, logvar)
        g_loss = self.bce_loss(self.disc(fake), self.valid)
        vae_loss = kld_loss + g_loss
        pred = self.clf(z)
        clf_loss = self.ce_loss(pred, labels)
        vae_clf_loss = vae_loss + clf_loss
        vae_clf_loss.backward()
        self.optimizerG.step()

        # train discriminator
        self.optimizerD.zero_grad()
        d_real_loss = self.bce_loss(self.disc(images), self.valid)
        d_fake_loss = self.bce_loss(self.disc(fake.detach()), self.fake)
        d_loss = (d_fake_loss + d_real_loss) / 2.0
        d_loss.backward()
        self.optimizerD.step()

        self.dashboard.add_trace_dict({'g_loss': g_loss.detach(),
                                       'd_loss': d_loss.detach(),
                                       'clf_loss': clf_loss.detach(),
                                       'kld_loss': kld_loss.detach()}, self.step)
        # self.dashboard.add_image_dict({'train_gen_image': gen_images,
        #                                 'train_image': images})
        loss = vae_clf_loss + d_loss
        return loss.detach(), None, None

    def eval_step(self, item):
        images, labels = item
        images = self.to_gpu(images)
        labels = self.to_gpu(labels)
        batch_size = images.size(0)
        self.update_label(batch_size)

        mu, logvar, fake, z = self.vae(images)

        # eval generator
        kld_loss = self.kld_loss(mu, logvar)
        g_loss = self.bce_loss(self.disc(fake), self.valid)
        pred = self.clf(z)
        clf_loss = self.ce_loss(pred, labels)
        vae_loss = kld_loss + g_loss
        vae_clf_loss = vae_loss + clf_loss

        # eval discriminator
        d_real_loss = self.bce_loss(self.disc(images), self.valid)
        d_fake_loss = self.bce_loss(self.disc(fake), self.fake)
        d_loss = (d_real_loss + d_fake_loss) / 2.0

        self.dashboard.add_image_dict({'eval_fake': fake,
                                       'eval_image': images})

        loss = vae_clf_loss + d_loss
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
