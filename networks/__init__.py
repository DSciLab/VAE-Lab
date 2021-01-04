import importlib
from torch import nn


def find_mod(mod):
    for sub_mod_name in dir(mod):
        sub_mod = getattr(mod, sub_mod_name)
        if issubclass(sub_mod, nn.Module) and sub_mod is not nn.Module:
            return sub_mod


def get_classifier(opt):
    try:
        mod = importlib.import_module(f'networks.classifier.{opt.classifier}')
    except ModuleNotFoundError as e:
        raise RuntimeError(
            f'Unrecognized classifier {opt.classifier}')

    classifier_cls = find_mod(mod)
    return classifier_cls


def get_vae(opt):
    try:
        mod = importlib.import_module(f'networks.vae.{opt.vae}')
    except ModuleNotFoundError as e:
        raise RuntimeError(
            f'Unrecognized vae {opt.vae}')

    vae_cls = find_mod(mod)
    return vae_cls
