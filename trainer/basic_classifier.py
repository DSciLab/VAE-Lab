from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from mlutils import Trainer
from networks.classifier.basic import BasicClassifier


class ClassificationTrainer(Trainer):
    def __init__(self, opt):
        super().__init__(opt)
        model = BasicClassifier(opt)
        self.optimizer = SGD(model.parameters(),
                            lr=opt.lr,
                            momentum=0.9)
        self.scheduler = MultiStepLR(self.optimizer,
                            milestones=[50, 80],
                            gamma=0.1)
        self.model = self.to_gpu(model)
        self.loss_fn = nn.CrossEntropyLoss()

    def train_step(self, item):
        images, labels = item
        images = self.to_gpu(images)
        labels = self.to_gpu(labels)

        self.optimizer.zero_grad()
        proba, logit = self.model(images)
        loss = self.loss_fn(logit, labels)
        loss.backward()
        self.optimizer.step()

        return loss.detach(), proba, labels

    def eval_step(self, item):
        images, labels = item
        images = self.to_gpu(images)
        labels = self.to_gpu(labels)

        proba, logit = self.model(images)
        loss = self.loss_fn(logit, labels)

        return loss.detach(), proba, labels

    def inference(self, image):
        pass

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self):
        pass

    def on_training_begin(self):
        pass

    def on_training_end(self):
        pass
