import pretrainedmodels
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torchvision
from efficientnet_pytorch import EfficientNet
from functools import partial
from catalyst.dl.utils import load_checkpoint
from config.base import load_config
from .head import CustomHead, MultiSampleDropout, Identity, MultiHead, AuxHead
from . pooling import GeM

MODEL_CONFIG = dict(
    resnet18=dict(
        source=torchvision.models,
        pretrained_arg=True,
        fix_ave_pool=True,
        fc_name='fc'
    ),
    resnet34=dict(
        source=torchvision.models,
        pretrained_arg=True,
        fix_ave_pool=True,
        fc_name='fc'
    ),
    resnet50=dict(
        source=torchvision.models,
        pretrained_arg=True,
        fix_ave_pool=True,
        fc_name='fc'
    ),
    se_resnext50_32x4d=dict(
        source=pretrainedmodels,
        pretrained_arg='imagenet',
        fix_ave_pool=True,
        fc_name='last_linear'
    ),
    se_resnext101_32x4d=dict(
        source=pretrainedmodels,
        pretrained_arg='imagenet',
        fix_ave_pool=True,
        fc_name='last_linear'
    ),
    efficientnet_b0=dict(
        source=EfficientNet.from_pretrained,
        pretrained_arg=None,
        fix_ave_pool=False,
        fc_name='_fc'
    ),
    resnext101_32x16d_wsl=dict(
        source=partial(torch.hub.load, 'facebookresearch/WSL-Images'),
        pretrained_arg=None,
        fix_ave_pool=False,
        fc_name='fc'
    ),
)

FC_MODULES = dict(
    linear=nn.Linear,
    custom_head=CustomHead,
    multi_sample_dropout=MultiSampleDropout,
    identity=Identity,
    multi_head=MultiHead,
    aux_head=AuxHead
)

POOLING_MODULES = dict(
    adaptive=nn.AdaptiveAvgPool2d(1),
    gem=GeM(),
)

class MyModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.cfg = MODEL_CONFIG[config.model.name]
        self.net = getattr(self.cfg['source'], config.model.name)
        if self.cfg['pretrained_arg']:
            self.net = self.net(pretrained=self.cfg['pretrained_arg'])
        if self.cfg['fix_ave_pool']:
            self.net.avg_pool = nn.AdaptiveAvgPool2d(1)
        if config.model.pooling == 'gem':
            self.net.avg_pool = GeM()

        fc = getattr(self.net, self.cfg['fc_name'])
        fc = FC_MODULES[config.model.fc](fc.in_features, config.data.num_classes)
        setattr(self.net, self.cfg['fc_name'], fc)

    def fresh_params(self):
        return getattr(self.net, self.cfg['fc_name']).parameters()

    def base_params(self):
        params = []
        for name, param in self.net.named_parameters():
            if self.cfg['fc_name'] not in name:
                params.append(param)
        return params

    def forward(self, x):
        return self.net(x)

class MultiModels:
    def __init__(self, models):
        self.models = models

    def __call__(self, x):
        res = []
        x = x.cuda()
        with torch.no_grad():
            for m in self.models:
                res.append(m(x))
        res = torch.stack(res)
        return torch.mean(res, dim=0)


def get_model(config, checkpoint_path=None):
    model = MyModel(config)
    if checkpoint_path:
        checkpoint = load_checkpoint(checkpoint_path)
        print('load model from:', checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])    
    return model