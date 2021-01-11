import importlib
import inspect
from torch import nn


def get_classifier(opt):
    try:
        mod = importlib.import_module(f'networks.classifier.{opt.classifier}')
    except ModuleNotFoundError as e:
        raise RuntimeError(
            f'Unrecognized classifier {opt.classifier}')

    return mod


def get_vae(opt):
    try:
        mod = importlib.import_module(f'networks.vae.{opt.vae}')
    except ModuleNotFoundError as e:
        raise RuntimeError(
            f'Unrecognized vae {opt.vae}')

    return mod
