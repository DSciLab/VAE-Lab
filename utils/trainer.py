from models.classifier import BasicClassifier
from mlutils import Trainer


class VAETrainer(Trainer):
    def __init__(self, opt):
        super().__init__()

    def train_step(self, item):
        pass

    def eval_step(self, item):
        pass

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self):
        pass

    def on_training_begin(self):
        pass

    def on_training_end(self):
        pass
