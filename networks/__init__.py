from _typeshed import ReadableBuffer
from .classifier.basic import BasicClassifier
from .vae.basic import BasicVAE


def get_classifier(opt):
    if opt.classifier == 'basic':
        return BasicClassifier
    else:
        raise RuntimeError(
            f'Unrecognized classifier {opt.classifier}')

def get_vae(opt):
    if opt.vae == 'vae':
        return BasicVAE
    else:
        raise RuntimeError(
            f'Unrecognized vae {opt.vae}')
