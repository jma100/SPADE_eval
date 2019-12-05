import argparse
import contextlib
import oyaml as yaml

import torch
import pyro
import visdom

from . import options

######################
# Namespace attributes
######################

overwriting = False       # If True, disable the setattr check below to allow overwritting


@contextlib.contextmanager
def overwrite(value=True):
    global overwriting
    old_value = overwriting
    overwriting = value
    yield
    overwriting = old_value


############################
# Printing & Dumping
############################

# We add custom yaml representer yaml_module_representer for all
# `nn.Module`s (otherwise a weird error will be thrown) and
# `torch/pyro.distribution.Distribution`s (for nicer display, i.e., not
# showing serialized tensors).
#
# Yet adding one a superclass doesn't affect dumping yaml for subclasses. So
# we have this set checking if we have done so for every nn.Module we have
# seen.
yaml_specialized_superclasses = (torch.nn.Module, torch.distributions.Distribution,
                                 pyro.distributions.Distribution, visdom.Visdom,
                                 torch.utils.data.DataLoader)
yaml_specialized_classes = set()


def yaml_module_representer(dumper, data):
    data_repr = repr(data)
    if '\n' in data_repr:
        data_repr = object.__repr__(data)
    return dumper.represent_scalar('!custom/{}.{}'.format(data.__module__, data.__class__.__name__), data_repr)


def display_yaml(data):
    return yaml.dump(data, default_flow_style=False, indent=4, width=120)


#################
# Namespace class
#################

class Namespace(argparse.Namespace):
    def __init__(self, local_yaml_display_config):
        super().__init__()
        self.local_yaml_display_config = local_yaml_display_config

    @staticmethod
    def _setattr_to_namespace(namespace, display_yaml_attr_name, key, value):
        if hasattr(namespace, key) and not overwriting:
            raise RuntimeError("cannot overwrite config option '{}'".format(key))

        display_yaml = getattr(namespace, display_yaml_attr_name, {})
        if options.options_parsed() and key in display_yaml and not key.startswith('_'):
            # `display_yaml` being `None` means that the config is not yet
            # parsed, and that things are still moving.

            # Somehow yaml-ing a plain torch.nn.Module or a torch.distribution.Distribution
            # breaks with a weird error similar to https://github.com/pytorch/pytorch/issues/11172
            if isinstance(value, yaml_specialized_superclasses) and value.__class__ not in yaml_specialized_classes:
                yaml.add_representer(value.__class__, yaml_module_representer)
                yaml_specialized_classes.add(value.__class__)

            display_yaml[key] = value

        return super(namespace.__class__, namespace).__setattr__(key, value)

    def __setattr__(self, key, value):
        return Namespace._setattr_to_namespace(self, 'local_yaml_display_config', key, value)

    def __repr__(self):
        if hasattr(self, 'yaml_display_config'):
            return display_yaml(self.yaml_display_config)
        else:
            return super().__repr__()
