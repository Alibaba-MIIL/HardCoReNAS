import logging
import os
from collections import OrderedDict
from copy import deepcopy
from typing import Callable

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from external.distributed_manager import DistributedManager
from timm.models.features import FeatureListNet, FeatureHookNet
from timm.models.layers.conv2d_same import Conv2dSame


def load_state_dict(checkpoint_path, use_ema=False):
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict_key = 'state_dict'
        if isinstance(checkpoint, dict):
            if use_ema and 'state_dict_ema' in checkpoint:
                state_dict_key = 'state_dict_ema'

        if state_dict_key and state_dict_key in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint[state_dict_key].items():
                # strip `module.` prefix
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v

            state_dict = new_state_dict

        else:
            state_dict = checkpoint

        if DistributedManager.is_master():
            logging.info("Loaded {} from checkpoint '{}'".format(state_dict_key, checkpoint_path))

        return state_dict

    else:
        if DistributedManager.is_master():
            logging.error("No checkpoint found at '{}'".format(checkpoint_path))

        raise FileNotFoundError()


def load_checkpoint(model, checkpoint_path, use_ema=False, strict=True):
    if checkpoint_path.startswith('http'):
        state_dict = model_zoo.load_url(checkpoint_path, progress=True, map_location='cpu')
        model.load_state_dict(state_dict, strict=strict)
        return

    state_dict = load_state_dict(checkpoint_path, use_ema)
    try:
        model.load_state_dict(state_dict, strict=strict)
    except:
        if 'fc.weight' in state_dict:
            state_dict.pop('fc.weight')
            state_dict.pop('fc.bias')

        elif 'fc_linear' in state_dict:
            state_dict.pop('fc_linear.weight')
            state_dict.pop('fc_linear.bias')

        else:
            state_dict.pop('classifier.weight')
            state_dict.pop('classifier.bias')

        model.load_state_dict(state_dict, strict=strict)
        if 'fc.weight' in model.state_dict():
            model.fc.weight.data.normal_(mean=0.0, std=0.01)
            model.fc.bias.data.zero_()

        elif 'fc_linear' in model.state_dict():
            model.fc_linear.weight.data.normal_(mean=0.0, std=0.01)
            model.fc_linear.bias.data.zero_()



def resume_checkpoint(model, checkpoint_path):
    other_state = {}
    resume_epoch = None
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                name = k[7:] if k.startswith('module') else k
                new_state_dict[name] = v

            model.load_state_dict(new_state_dict)
            if 'optimizer' in checkpoint:
                other_state['optimizer'] = checkpoint['optimizer']

            if 'amp' in checkpoint:
                other_state['amp'] = checkpoint['amp']

            if 'epoch' in checkpoint:
                resume_epoch = checkpoint['epoch']
                if 'version' in checkpoint and checkpoint['version'] > 1:
                    resume_epoch += 1  # start at the next epoch, old checkpoints incremented before save

            if DistributedManager.is_master():
                logging.info("Loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))

        else:
            model.load_state_dict(checkpoint)
            if DistributedManager.is_master():
                logging.info("Loaded checkpoint '{}'".format(checkpoint_path))

        return other_state, resume_epoch

    else:
        if DistributedManager.is_master():
            logging.error("No checkpoint found at '{}'".format(checkpoint_path))

        raise FileNotFoundError()


def load_pretrained(model, cfg=None, num_classes=1000, in_chans=3, filter_fn=None, strict=True):
    if cfg is None:
        cfg = getattr(model, 'default_cfg')

    if cfg is None or 'url' not in cfg or not cfg['url']:
        logging.warning("Pretrained model URL is invalid, using random initialization.")
        return

    state_dict = model_zoo.load_url(cfg['url'], progress=False, map_location='cpu')

    if in_chans == 1:
        conv1_name = cfg['first_conv']
        logging.info('Converting first conv (%s) from 3 to 1 channel' % conv1_name)
        conv1_weight = state_dict[conv1_name + '.weight']
        state_dict[conv1_name + '.weight'] = conv1_weight.sum(dim=1, keepdim=True)

    elif in_chans != 3:
        assert False, "Invalid in_chans for pretrained weights"

    classifier_name = cfg['classifier']
    if num_classes == 1000 and cfg['num_classes'] == 1001:
        # Special case for imagenet trained models with extra background class in pretrained weights
        classifier_weight = state_dict[classifier_name + '.weight']
        state_dict[classifier_name + '.weight'] = classifier_weight[1:]
        classifier_bias = state_dict[classifier_name + '.bias']
        state_dict[classifier_name + '.bias'] = classifier_bias[1:]

    elif num_classes != cfg['num_classes']:
        # Completely discard fully connected for all other differences between pretrained and created model
        del state_dict[classifier_name + '.weight']
        del state_dict[classifier_name + '.bias']
        strict = False

    if filter_fn is not None:
        state_dict = filter_fn(state_dict)

    model.load_state_dict(state_dict, strict=strict)


def extract_layer(model, layer):
    layer = layer.split('.')
    module = model
    if hasattr(model, 'module') and layer[0] != 'module':
        module = model.module

    if not hasattr(model, 'module') and layer[0] == 'module':
        layer = layer[1:]

    for l in layer:
        if hasattr(module, l):
            if not l.isdigit():
                module = getattr(module, l)

            else:
                module = module[int(l)]

        else:
            return module

    return module


def set_layer(model, layer, val):
    layer = layer.split('.')
    module = model
    if hasattr(model, 'module') and layer[0] != 'module':
        module = model.module

    lst_index = 0
    module2 = module
    for l in layer:
        if hasattr(module2, l):
            if not l.isdigit():
                module2 = getattr(module2, l)

            else:
                module2 = module2[int(l)]

            lst_index += 1

    lst_index -= 1
    for l in layer[:lst_index]:
        if not l.isdigit():
            module = getattr(module, l)

        else:
            module = module[int(l)]

    l = layer[lst_index]
    setattr(module, l, val)


def adapt_model_from_string(parent_module, model_string):
    separator = '***'
    state_dict = {}
    lst_shape = model_string.split(separator)
    for k in lst_shape:
        k = k.split(':')
        key = k[0]
        shape = k[1][1:-1].split(',')
        if shape[0] != '':
            state_dict[key] = [int(i) for i in shape]

    new_module = deepcopy(parent_module)
    for n, m in parent_module.named_modules():
        old_module = extract_layer(parent_module, n)
        if isinstance(old_module, nn.Conv2d) or isinstance(old_module, Conv2dSame):
            if isinstance(old_module, Conv2dSame):
                conv = Conv2dSame

            else:
                conv = nn.Conv2d

            s = state_dict[n + '.weight']
            in_channels = s[1]
            out_channels = s[0]
            g = 1
            if old_module.groups > 1:
                in_channels = out_channels
                g = in_channels

            new_conv = conv(
                in_channels=in_channels, out_channels=out_channels, kernel_size=old_module.kernel_size,
                bias=old_module.bias is not None, padding=old_module.padding, dilation=old_module.dilation,
                groups=g, stride=old_module.stride)

            set_layer(new_module, n, new_conv)

        if isinstance(old_module, nn.BatchNorm2d):
            new_bn = nn.BatchNorm2d(
                num_features=state_dict[n + '.weight'][0], eps=old_module.eps, momentum=old_module.momentum,
                affine=old_module.affine, track_running_stats=True)
            set_layer(new_module, n, new_bn)

        if isinstance(old_module, nn.Linear):
            new_fc = nn.Linear(
                in_features=state_dict[n + '.weight'][1], out_features=old_module.out_features,
                bias=old_module.bias is not None)
            set_layer(new_module, n, new_fc)

    new_module.eval()
    parent_module.eval()

    return new_module


def adapt_model_from_file(parent_module, model_variant):
    adapt_file = os.path.join(os.path.dirname(__file__), 'pruned', model_variant + '.txt')
    with open(adapt_file, 'r') as f:
        return adapt_model_from_string(parent_module, f.read().strip())


def build_model_with_cfg(
        model_cls: Callable,
        variant: str,
        pretrained: bool,
        default_cfg: dict,
        model_cfg: dict = None,
        feature_cfg: dict = None,
        pretrained_strict: bool = True,
        pretrained_filter_fn: Callable = None,
        **kwargs):
    pruned = kwargs.pop('pruned', False)
    features = False
    feature_cfg = feature_cfg or {}

    if kwargs.pop('features_only', False):
        features = True
        feature_cfg.setdefault('out_indices', (0, 1, 2, 3, 4))
        if 'out_indices' in kwargs:
            feature_cfg['out_indices'] = kwargs.pop('out_indices')

    model = model_cls(**kwargs) if model_cfg is None else model_cls(cfg=model_cfg, **kwargs)
    model.default_cfg = deepcopy(default_cfg)

    if pruned:
        model = adapt_model_from_file(model, variant)

    # for classification models, check class attr, then kwargs, then default to 1k, otherwise 0 for feats
    num_classes_pretrained = 0 if features else getattr(model, 'num_classes', kwargs.get('num_classes', 1000))
    if pretrained:
        load_pretrained(
            model,
            num_classes=num_classes_pretrained, in_chans=kwargs.get('in_chans', 3),
            filter_fn=pretrained_filter_fn, strict=pretrained_strict)

    if features:
        feature_cls = FeatureListNet
        if 'feature_cls' in feature_cfg:
            feature_cls = feature_cfg.pop('feature_cls')
            if isinstance(feature_cls, str):
                feature_cls = feature_cls.lower()
                if 'hook' in feature_cls:
                    feature_cls = FeatureHookNet
                else:
                    assert False, f'Unknown feature class {feature_cls}'
        model = feature_cls(model, **feature_cfg)

    return model


def fuse_bn_to_conv(bn_layer, conv_layer):
    # print('bn fuse')
    bn_st_dict = bn_layer.state_dict()
    conv_st_dict = conv_layer.state_dict()

    # BatchNorm params
    eps = bn_layer.eps
    mu = bn_st_dict['running_mean']
    var = bn_st_dict['running_var']
    gamma = bn_st_dict['weight']

    if 'bias' in bn_st_dict:
        beta = bn_st_dict['bias']

    else:
        beta = torch.zeros(gamma.size(0)).float().to(gamma.device)

    # Conv params
    W = conv_st_dict['weight']
    if 'bias' in conv_st_dict:
        bias = conv_st_dict['bias']

    else:
        bias = torch.zeros(W.size(0)).float().to(gamma.device)

    denom = torch.sqrt(var + eps)
    b = beta - gamma.mul(mu).div(denom)
    A = gamma.div(denom)
    bias *= A
    A = A.expand_as(W.transpose(0, -1)).transpose(0, -1)

    W.mul_(A)
    bias.add_(b)

    conv_layer.weight.data.copy_(W)
    if conv_layer.bias is None:
        conv_layer.bias = torch.nn.Parameter(bias)

    else:
        conv_layer.bias.data.copy_(bias)


def extract_layers(model):
    list_layers = []
    for n, p in model.named_modules():
        list_layers.append(n)

    return list_layers


def compute_next_bn(layer_name, resnet):
    list_layer = extract_layers(resnet)
    assert layer_name in list_layer
    if layer_name == list_layer[-1]:
        return None

    next_bn = list_layer[list_layer.index(layer_name) + 1]
    if extract_layer(resnet, next_bn).__class__.__name__ == 'BatchNorm2d':
        return next_bn

    return None


def fuse_bn(model):
    for n, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            next_bn = compute_next_bn(n, model)
            if next_bn is not None:
                next_bn_ = extract_layer(model, next_bn)
                fuse_bn_to_conv(next_bn_, m)
                set_layer(model, next_bn, nn.Identity())

    return model