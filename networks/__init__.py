import importlib
import inspect
from torch import nn


def find_mod(mod, match_name):
    for sub_mod_name in dir(mod):
        sub_mod = getattr(mod, sub_mod_name)
        if inspect.isclass(sub_mod) and \
            issubclass(sub_mod, nn.Module) \
            and sub_mod is not nn.Module \
            and match_name.lower() in sub_mod_name.lower():
            return sub_mod


def get_classifier(opt):
    try:
        mod = importlib.import_module(f'networks.classifier.{opt.classifier}')
    except ModuleNotFoundError as e:
        raise RuntimeError(
            f'Unrecognized classifier {opt.classifier}')

    classifier_cls = find_mod(mod, opt.classifier)
    return classifier_cls


def get_vae(opt):
    try:
        mod = importlib.import_module(f'networks.vae.{opt.vae}')
    except ModuleNotFoundError as e:
        raise RuntimeError(
            f'Unrecognized vae {opt.vae}')

    vae_cls = find_mod(mod, opt.vae)
    return vae_cls
