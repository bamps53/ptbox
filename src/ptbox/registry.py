from typing import Callable

import torch
from omegaconf import DictConfig, ListConfig

# from https://github.com/open-mmlab/mmcv/blob/master/mmcv/utils/registry.py
class Registry:
    """A registry to map strings to classes.
    Args:
        name (str): Registry name.
    """

    def __init__(self, name: str):
        self._name = name
        self._obj_dict = dict()

    def __repr__(self):
        format_str = self.__class__.__name__
        format_str += f'(name={self._name}, items={list(self._obj_dict.keys())})'
        return format_str

    @property
    def name(self):
        return self._name

    @property
    def obj_dict(self):
        return self._obj_dict

    def get(self, key: str):
        return self._obj_dict.get(key, None)

    def register(self, obj: Callable):
        """Register a callable object.
        Args:
            obj: callable object to be registered
        """
        if not callable(obj):
            raise ValueError(f'object must be callable')

        obj_name = obj.__name__
        if obj_name in self._obj_dict:
            print(f'{obj_name} is already registered in {self.name}')
            # raise KeyError(f'{obj_name} is already registered in {self.name}')

        self._obj_dict[obj_name] = obj
        return obj


def build_from_config(config: dict, registry: Registry, default_args: dict = None):
    """Build a callable object from configuation dict.
    Args:
        config (dict): Configuration dict. It should contain the key "name".
        registry (:obj:`Registry`): The registry to search the name from.
        default_args (dict, optional): Default initialization argments.
    """
    assert (isinstance(config, dict) or isinstance(config, DictConfig)) and 'name' in config, print(config)
    assert isinstance(
        default_args, dict) or default_args is None, print(default_args)

    name = config['name']
    name = name.replace('-', '_')
    obj = registry.get(name)
    if obj is None:
        raise KeyError(f'{name} is not in the {registry.name} registry')

    args = dict()
    if default_args is not None:
        args.update(default_args)
    if 'params' in config:
        args.update(config['params'])
    return obj(**args)


def build_from_config_list(configs: list, registry: Registry, default_args: dict = None):
    """Build a callable object from configuation list.
    Args:
        configs (list): List of configuration dict.
        registry (:obj:`Registry`): The registry to search the name from.
        default_args (dict, optional): Default initialization argments.
    """
    if not (isinstance(configs, list) or isinstance(configs, ListConfig)):
        return []  # and 'name' in config

    objects = []
    for c in configs:
        if 'take_default_args' in c.keys():
            c.pop('take_default_args')
            obj = build_from_config(c, registry, default_args)
        else:
            obj = build_from_config(c, registry)
        objects.append(obj)
    return objects


def build_from_config_dict(configs: dict, registry: Registry, default_args: dict = None):
    """Build a callable object from configuation dict.
    Args:
        configs (dict): Dict of configuration dict.
        registry (:obj:`Registry`): The registry to search the name from.
        default_args (dict, optional): Default initialization argments.
    """
    if not (isinstance(configs, dict) or isinstance(configs, DictConfig)):
        return {}  # and 'name' in config

    objects = {}
    for name, c in configs.items():
        if 'take_default_args' in c.keys():
            obj = build_from_config(c, registry, default_args)
        else:
            obj = build_from_config(c, registry)
        objects[name] = obj
    return objects


MODELS = Registry('models')
LOSSES = Registry('losses')
OPTIMIZERS = Registry('optimizers')
SCHEDULERS = Registry('schedulers')
DATASETS = Registry('datasets')
CALLBACKS = Registry('callbacks')
METRICS = Registry('metrics')

# losses
for k, v in torch.nn.__dict__.items():
    if 'Loss' in k:
        if callable(v) and isinstance(v, type):
            LOSSES.register(v)

# optimizers
for k, v in torch.optim.__dict__.items():
    if callable(v):
        OPTIMIZERS.register(v)

# schedulers
for k, v in torch.optim.lr_scheduler.__dict__.items():
    if callable(v):
        SCHEDULERS.register(v)

# callbacks
from catalyst.dl import callbacks
for k, v in callbacks.__dict__.items():
    if 'Callback' in k or 'Logger' in k:
        if callable(v) and isinstance(v, type):
            CALLBACKS.register(v)
