import logging
import math
import os.path
import pickle
import time
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from external.distributed_manager import DistributedManager
from timm.models.layers.adaptive_avgmax_pool import SelectAdaptivePool2d
from timm.models.layers.conv2d_same import Conv2dSame

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as DDP
    from apex.parallel import convert_syncbn_model

    has_apex = True
except ImportError:
    from torch.nn.parallel import DistributedDataParallel as DDP


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


def extract_conv_layers(model, conv1d=True):
    list_conv = []
    for n, p in model.named_modules():
        if isinstance(p, nn.Conv2d) or (isinstance(p, nn.Conv1d) and conv1d):
            list_conv.append(n)
    return list_conv


def extract_layers(model):
    list_layers = []
    for n, p in model.named_modules():
        list_layers.append(n)
    return list_layers


def compute_stat_svd(model, loss, layers_list, data_loader, taylor=False):
    mean_over_batch = {}
    num_batch = {}
    sv = {}

    class write_state_forward_svd:
        def __init__(self, layer):
            self.layer = layer

        def __call__(self, module, input, output):
            nonlocal mean_over_batch
            nonlocal num_batch
            layer = self.layer
            input_ = output  # input[0]
            s = input_.shape
            input_reshaped = input_.view(s[0], s[1], -1)
            covar_mat_batch = torch.matmul(input_reshaped, input_reshaped.transpose(2, 1))
            covar_mat_mean_over_batch = 1 / s[0] * covar_mat_batch.sum(dim=0).cpu()
            if not layer in num_batch:
                mean_over_batch[layer] = covar_mat_mean_over_batch
                num_batch[layer] = torch.Tensor([1])
            else:
                num_batch[layer] = (num_batch[layer] + 1)
                mean_over_batch[layer] = mean_over_batch[layer]
                mean_over_batch[layer] = mean_over_batch[layer] * (
                    1 - 1 / num_batch[layer]) + covar_mat_mean_over_batch / num_batch[layer]

    hook_forward = {}
    for layer in layers_list:
        layer_ = extract_layer(model, layer)
        hook_forward[layer] = layer_.register_forward_hook(write_state_forward_svd(layer))
    model.eval()

    if taylor:
        model.zero_grad()
    for batch_idx, (input, target) in enumerate(data_loader):
        input, target = input.cuda(), target.cuda()
        if taylor:
            l = loss(model(input), target)
            l.backward()
        else:
            with torch.no_grad():
                loss(model(input), target)

    for layer in layers_list:
        hook_forward[layer].remove()
        s, _ = torch.symeig(mean_over_batch[layer], eigenvectors=False)
        sv[layer] = s
    if taylor:
        taylor_per_layer = {}
        for layer in layers_list:
            layer_ = extract_layer(model, layer)
            taylor_per_layer[layer] = torch.sum(torch.abs(layer_.weight * layer_.weight.grad), dim=[2, 3])
        return taylor_per_layer, sv
    return sv


def compute_taylor(model, loss, layers_list, data_loader, len_data_loader=None, local_rank=0):
    model.zero_grad()
    model.eval()
    if len_data_loader is None:
        len_data_loader = len(data_loader)
    for batch_idx, (input, target) in enumerate(data_loader):
        if batch_idx == len_data_loader:
            break
        input, target = input.cuda(), target.cuda()
        output = model(input)
        l = loss(output, target) / len_data_loader
        l.backward()
        if batch_idx % 10 == 0 and local_rank == 0:
            logging.info(f"Computing gradient for validation: Done {batch_idx}/{len_data_loader}")
    taylor_per_layer = {}
    model.cpu()
    for layer in layers_list:
        layer_ = extract_layer(model, layer)
        taylor_per_layer[layer] = torch.sum(torch.abs(layer_.weight * layer_.weight.grad), dim=[1, 2, 3])
    model.cuda()
    return taylor_per_layer


def compute_taylor2(model, loss, layers_list, data_loader, len_data_loader=None, local_rank=0):
    model.zero_grad()
    model.eval()
    if len_data_loader is None:
        len_data_loader = len(data_loader)
    taylor_per_layer = {}
    for layer in layers_list:
        taylor_per_layer[layer] = 0
    model.cuda()
    logging.info(f"Using Taylor abs, initializing zero gradient for model")
    model.zero_grad()
    for batch_idx, (input, target) in enumerate(data_loader):
        if batch_idx == len_data_loader:
            break
        input, target = input.cuda(), target.cuda()
        output = model(input)
        l = loss(output, target) / len_data_loader
        l.backward()
        if batch_idx % 10 == 0 and local_rank == 0:
            logging.info(f"Computing gradient for validation: Done {batch_idx}/{len_data_loader}")
        for layer in layers_list:
            layer_ = extract_layer(model, layer)
            taylor_per_layer[layer] += torch.abs(torch.sum(layer_.weight * layer_.weight.grad, dim=[1, 2, 3]))
            # layer_.weight.grad *= 0
        model.zero_grad()
    return taylor_per_layer


def extract_statistics_from_model(model, loss, data_loader):
    list_conv = extract_conv_layers(model)
    taylor_per_layer, sv = compute_stat_svd(model, loss, list_conv, data_loader, taylor=True)
    return taylor_per_layer, sv


def compute_time_per_layer(model, input_size, batch_size=512):
    time_total, dict_time, _ = measure_time(model, iterations=10, input_size=input_size, batch_size=batch_size)
    layers_list = extract_conv_layers(model)
    time_layer = {}
    for layer in layers_list:
        output_channel = extract_layer(model, layer).weight.shape[1]
        time_layer[layer] = [int(1e7 * dict_time[layer] / output_channel), int(1e7 * dict_time[layer])]
    return time_layer


def compute_macs_per_layer(model, input_size, take_in_channel_saving=False):
    if not (isinstance(input_size, list) or isinstance(input_size, tuple)):
        input_size = [3, input_size, input_size]
    input_tensor = torch.randn(1, input_size[0], input_size[1], input_size[2], requires_grad=False)
    if hasattr(model, 'device'):
        input_tensor.to(model.device)
    layers_list = extract_conv_layers(model)
    macs = {}
    hook_forward = {}

    class write_mac_forward:
        def __init__(self, layer):
            self.layer = layer

        def __call__(self, module, input, output):
            nonlocal macs
            layer = self.layer
            input_ = input[0]
            s = input_.shape
            num_out_channel = output.shape[1]
            num_channel = s[1]
            g = module.groups
            if isinstance(module, nn.Conv2d):
                W = (s[2] - module.weight.shape[2] + 2 * module.padding[0]) / module.stride[0] + 1
                H = (s[3] - module.weight.shape[3] + 2 * module.padding[1]) / module.stride[1] + 1
                size_kernel = module.weight.shape[2] * module.weight.shape[3]
            else:
                W = (s[2] - module.weight.shape[2] + 2 * module.padding[0]) / module.stride[0] + 1
                H = 1
                size_kernel = module.weight.shape[2]
            assert module.weight.shape[1] * g == num_channel and module.weight.shape[0] == num_out_channel
            mac_per_out_channel = int(module.weight.shape[1] * W * H * size_kernel)
            mac_per_in_channel = int(num_out_channel * W * H * size_kernel / g)
            macs[layer] = [mac_per_out_channel, mac_per_out_channel * num_out_channel, mac_per_in_channel]

    for layer in layers_list:
        layer_ = extract_layer(model, layer)
        hook_forward[layer] = layer_.register_forward_hook(write_mac_forward(layer))
    model.eval()
    # model.cuda()
    # input_tensor = input_tensor.cuda()
    model.cpu()
    model(input_tensor)
    for layer in layers_list:
        hook_forward[layer].remove()
        if take_in_channel_saving:
            model2 = model
            if isinstance(model, nn.DataParallel) or isinstance(model, nn.parallel.distributed.DistributedDataParallel):
                model2 = model.module
            if 'resnet' in model2.__class__.__name__.lower():
                next_layer = compute_next_layer_resnet(layer, model)
                if next_layer is not None:
                    macs[layer][0] += macs[next_layer][2]
            if 'efficient' in model2.__class__.__name__.lower():
                next_layer = compute_next_layer_effnet(layer, model)
                if next_layer is not None:
                    macs[layer][0] += macs[next_layer][2]

    return macs


def compute_flops(model, input_size, flops_conv=False):
    flops_all = compute_macs_per_layer(model, input_size)
    flops = 0
    flops_per_conv = {}
    for k, v in flops_all.items():
        flops += v[1]
        flops_per_conv[k] = v[1]

    if flops_conv:
        return flops, flops_per_conv
    return flops


def knapsack_dp_efficient(values, weights, capacity, return_all=False):
    check_inputs(values, weights, capacity)
    n_items = len(values)
    # first try to reduce the complexity of the problem by finding the GCD of the weights:
    gcd = weights[0]
    for p in weights:
        gcd = math.gcd(p, gcd)
    for i in range(len(weights)):
        weights[i] = int(weights[i] / gcd)
    capacity = int(capacity / gcd)
    table = [[0.0] * (capacity + 1) for _ in range(2)]
    print("Allocating binary array for dynamic programming indicator")
    keep = [[False] * (capacity + 1) for _ in range(n_items + 1)]
    print("Table allocated")
    for i in tqdm(range(1, n_items + 1), desc="Dynamic programming in progress"):
        wi = weights[i - 1]  # weight of current item
        vi = values[i - 1]  # value of current item
        index_old = (i - 1) % 2
        index_new = i % 2
        for w in range(0, capacity + 1):
            val1 = vi + table[index_old][w - wi]
            val2 = table[index_old][w]
            if (wi <= w) and (val1 > val2):
                table[index_new][w] = val1
                keep[i][w] = True
            else:
                table[index_new][w] = val2
    picks = []
    K = capacity
    for i in range(n_items, 0, -1):
        if keep[i][K] == True:
            picks.append(i)
            K -= weights[i - 1]

    picks.sort()
    picks = [x - 1 for x in picks]  # change to 0-index

    if return_all:
        max_val = table[n_items % 2][capacity]
        return picks, max_val
    return picks


def knapsack_greedy(values, weights, capacity, do_not_normalize=False):
    check_inputs(values, weights, capacity)
    n_items = len(values)
    # first try to reduce the complexity of the problem by finding the GCD of the weights:
    gcd = weights[0]
    for p in weights:
        gcd = math.gcd(p, gcd)
    for i in range(len(weights)):
        weights[i] = int(weights[i] / gcd)
    capacity = int(capacity / gcd)
    l = np.argsort([values[i] / weights[i] for i in range(n_items)])
    if do_not_normalize:
        l = np.argsort([values[i] for i in range(n_items)])

    total_capacity = 0
    for i in weights:
        total_capacity += i
    i = 0
    picks = []
    c = total_capacity
    while c > capacity and i < len(l):
        picks.append(l[i])
        c = c - weights[l[i]]
        i = i + 1

    picks = list(set(range(n_items)).difference(picks))
    picks.sort()
    return picks


def compute_next_layer_effnet(layer_name, effnet, use_se=True):
    list_layer = extract_conv_layers(effnet)
    if not use_se:
        list_layer = [x for x in list_layer if '.se.' not in x]
    assert layer_name in list_layer
    if layer_name == list_layer[-1]:
        return None
    next_layer = list_layer[list_layer.index(layer_name) + 1]
    return next_layer


def compute_next_downsample(layer_name, module):
    list_layer = extract_conv_layers(module)
    assert layer_name in list_layer
    is_last = compute_last_layer_of_module_resnet(layer_name, module) == layer_name
    if not is_last:
        return None
    next_layer = compute_next_layer_resnet(layer_name, module, downsample=False)
    if next_layer is None:
        return None
    next_d = compute_last_layer_of_module_resnet(next_layer, module, downsample=True)
    if 'downsample' in next_d:
        return next_d
    return None


def compute_next_layer_resnet(layer_name, resnet, downsample=False):
    list_layer = extract_conv_layers(resnet, conv1d=False)
    assert layer_name in list_layer
    if layer_name == list_layer[-1]:
        return None
    next_layer = list_layer[list_layer.index(layer_name) + 1]
    if not downsample and 'downsample' in list_layer[list_layer.index(layer_name) + 1]:
        return list_layer[list_layer.index(layer_name) + 2]
    return next_layer


def compute_last_layer_of_module_resnet(layer_name, resnet, downsample=False):
    list_layer = extract_conv_layers(resnet, conv1d=False)
    assert layer_name in list_layer
    module_extract = lambda x: '.'.join(x.split('.')[:2]) if x.split('.')[0] != 'module' else '.'.join(x.split('.')[:3])
    if layer_name == list_layer[-1]:
        return layer_name
    if not downsample and 'downsample' in layer_name:
        layer_name = list_layer[list_layer.index(layer_name) + 1]

    next_layer = list_layer[list_layer.index(layer_name) + 1]
    curr_module = module_extract(layer_name)
    next_module = module_extract(next_layer)
    while next_module == curr_module and ('downsample' not in next_layer or downsample):
        layer_name = next_layer
        if layer_name == list_layer[-1]:
            return layer_name
        next_layer = list_layer[list_layer.index(layer_name) + 1]
        curr_module = module_extract(layer_name)
        next_module = module_extract(next_layer)
    return layer_name


def compute_next_bn_resnet(layer_name, resnet):
    list_layer = extract_layers(resnet)
    assert layer_name in list_layer
    if layer_name == list_layer[-1]:
        return None
    next_bn = list_layer[list_layer.index(layer_name) + 1]
    assert extract_layer(resnet, next_bn).__class__.__name__ == 'BatchNorm2d'
    return next_bn


def check_inputs(values, weights, capacity):
    # check variable type
    assert (isinstance(values, list))
    assert (isinstance(weights, list))
    assert (isinstance(capacity, int))
    # check value type
    assert (all(isinstance(val, int) or isinstance(val, float) for val in values))
    assert (all(isinstance(val, int) for val in weights))
    # check validity of value
    assert (all(val >= 0 for val in weights))
    assert (len(values) == len(weights))
    assert (capacity > 0)


def extract_group_of_conv_effnet(model, prune_pwl=True, use_se=True):
    layers_list = extract_conv_layers(model)
    layers_list = [x for x in layers_list if 'blocks' in x and not 'blocks.0.' in x]
    if not use_se:
        layers_list = [x for x in layers_list if '.se.' not in x]
    layers_blocks = []
    for x in layers_list:
        splitted = x.split('.')
        ind = 1
        if splitted[0] == 'module':
            ind = 2
        group = int(splitted[ind]) - 1
        subgroup = int(splitted[ind + 1])
        if len(layers_blocks) <= group:
            layers_blocks.append([])
        if len(layers_blocks[group]) <= subgroup:
            layers_blocks[group].append([x])
        else:
            layers_blocks[group][subgroup].append(x)
    list_groups = []
    for block in layers_blocks:
        prefix = '.'.join(block[0][0].split('.')[:-2])
        curr_skip_con = []
        for index in range(len(block)):
            curr_prefix = f'{prefix}.{index}'
            if use_se:
                list_groups.append(curr_prefix + '.se.conv_reduce')
                list_groups.append(
                    [curr_prefix + '.conv_pw', curr_prefix + '.conv_dw', curr_prefix + '.se.conv_expand'])
            else:
                list_groups.append(
                    [curr_prefix + '.conv_pw', curr_prefix + '.conv_dw'])
            curr_skip_con.append(curr_prefix + '.conv_pwl')
        if prune_pwl:
            list_groups.append(curr_skip_con)
    return list_groups, layers_list


def extract_group_of_conv_resnet(model, prune_skip=True, prune_conv1=False):
    layers_list_init = extract_conv_layers(model)
    list_groups = []
    if prune_conv1:
        list_groups = [[x] for x in layers_list_init if x[:5] == 'conv1']
    layers_list = [x for x in layers_list_init if 'layer' in x and '.se.' not in x]
    layer_extract = lambda x: x.split('.')[0] if x.split('.')[0] != 'module' else x.split('.')[1]
    group_of_skip = {}
    for x in layers_list:
        last_layer_d = compute_last_layer_of_module_resnet(x, model, downsample=True)
        last_layer = compute_last_layer_of_module_resnet(x, model, downsample=False)
        is_downsample = 'downsample' in x
        is_last_element = (last_layer == x or last_layer_d == x)
        if not is_downsample and not is_last_element:
            list_groups.append(x)
        else:
            current_layer = layer_extract(x)
            if current_layer not in group_of_skip:
                group_of_skip[current_layer] = [x]
            else:
                group_of_skip[current_layer].append(x)
    if prune_skip:
        for k, v in group_of_skip.items():
            list_groups.append(v)
    if prune_conv1:
        layers_list = [x for x in layers_list_init if 'layer' in x and '.se.' not in x or x[:5] == 'conv1']
    return list_groups, layers_list


def compute_num_channels_per_layer_taylor(model, input_size, data_loader, pruning_ratio=0.5,
                                          loss=nn.CrossEntropyLoss().cuda(), len_data_loader=None, algo='greedy',
                                          taylor_file=None, local_rank=0, separator='***', prune_pwl=True,
                                          prune_skip=True, taylor_abs=False, prune_conv1=False, use_se=True,
                                          use_time=False, distributed=False, do_not_normalize=False, no_taylor=False):
    if not (isinstance(input_size, list) or isinstance(input_size, tuple)):
        input_size = [3, input_size, input_size]
    if use_time:
        macs_per_layer = compute_time_per_layer(model, input_size, batch_size=6 * data_loader.loader.batch_size)
        if distributed:
            grp = DistributedManager.grp
            ws = torch.distributed.get_world_size()
            for k, v in macs_per_layer.items():
                v1 = torch.tensor(v[0]).cuda()
                v2 = torch.tensor(v[1]).cuda()
                torch.distributed.all_reduce(v1, op=torch.distributed.ReduceOp.SUM, group=grp)
                torch.distributed.all_reduce(v2, op=torch.distributed.ReduceOp.SUM, group=grp)
                macs_per_layer[k] = [int(v1.item() / ws), int(v2.item() / ws)]
    else:
        macs_per_layer = compute_macs_per_layer(model, input_size)

    if 'resnet' in model.__class__.__name__.lower() or (
        hasattr(model, 'module') and 'resnet' in model.module.__class__.__name__.lower()):
        list_groups, list_conv = extract_group_of_conv_resnet(model, prune_skip=prune_skip, prune_conv1=prune_conv1)
    else:
        list_groups, list_conv = extract_group_of_conv_effnet(model, prune_pwl=prune_pwl, use_se=use_se)

    total_mac_all_net = 0
    for k, v in macs_per_layer.items():
        total_mac_all_net += v[1]
    # total_mac = total_mac_all_net
    if not no_taylor:
        if taylor_file is None:
            if not taylor_abs:
                taylor_per_layer = compute_taylor(model, loss, list_conv, data_loader, len_data_loader, local_rank)
            else:
                taylor_per_layer = compute_taylor2(model, loss, list_conv, data_loader, len_data_loader, local_rank)
        else:
            if not os.path.exists(taylor_file):
                taylor_per_layer = compute_taylor(model, loss, list_conv, data_loader, len_data_loader, local_rank)
                if local_rank == 0:
                    with open(taylor_file, "wb") as fp:  # pickling
                        pickle.dump(taylor_per_layer, fp)
                    logging.info(f"Taylor per layer list saved in {taylor_file}")
            else:
                with open(taylor_file, "rb") as fp:  # Unpickling
                    taylor_per_layer = pickle.load(fp)
                if local_rank == 0:
                    logging.info(f"Taylor per layer loaded from {taylor_file}")
    else:
        taylor_per_layer = {}
        for layer in list_conv:
            layer_ = extract_layer(model, layer)
            taylor_per_layer[layer] = torch.sqrt(torch.sum(layer_.weight ** 2, dim=[1, 2, 3]))
    list_channel_name = []
    list_channel_cost = []
    list_channel_value = []
    total_mac = 0
    for layer_groups in list_groups:
        if not isinstance(layer_groups, list):
            layer = layer_groups
            macs = macs_per_layer[layer][0]
            if layer not in taylor_per_layer:
                taylor = taylor_per_layer['module.' + layer]
            else:
                taylor = taylor_per_layer[layer]
        else:
            macs = 0
            taylor = 0
            for layer in layer_groups:
                macs += macs_per_layer[layer][0]
                if layer not in taylor_per_layer:
                    taylor += taylor_per_layer['module.' + layer]
                else:
                    taylor += taylor_per_layer[layer]

            layer = separator.join(layer_groups)
        for i in range(taylor.shape[0]):
            key = layer + '.' + str(i)
            list_channel_name.append(key)
            list_channel_cost.append(macs)
            list_channel_value.append(taylor[i].item())
            total_mac += macs
    capacity = int(total_mac * (1 - pruning_ratio))
    if algo == 'greedy':
        list_channel_to_keep = knapsack_greedy(list_channel_value, list_channel_cost, capacity,
                                               do_not_normalize=do_not_normalize)
    else:
        list_channel_to_keep = knapsack_dp_efficient(list_channel_value, list_channel_cost, capacity)
    list_channel_to_prune = list(set(range(len(list_channel_value))).difference(set(list_channel_to_keep)))
    mac_saved = 0
    for i in list_channel_to_prune:
        l = list_channel_name[i]
        if separator not in l:
            index = l.rfind('.')
            layer = l[:index]
            mac_saved += macs_per_layer[layer][0]
        else:
            index = l.rfind('.')
            l = l[:index]
            l = l.split(separator)
            for layer in l:
                mac_saved += macs_per_layer[layer][0]
    # logging.info(f'total macs from layer to prune: {total_mac}, estimated mac to prune: {100*mac_saved/total_mac} %')
    # logging.info(
    #     f'total macs from entire network : {total_mac_all_net}, estimated mac to prune: {100*mac_saved/total_mac_all_net} %')
    return [list_channel_name[i] for i in list_channel_to_prune]


def redesign_module_efnet(module, list_channel_to_prune, use_amp=False, distributed=False, local_rank=0,
                          input_size=224, separator='***', use_se=True):
    is_parallel = False
    if isinstance(module, DDP) or isinstance(module, nn.DataParallel) or isinstance(module,
                                                                                    nn.parallel.DistributedDataParallel):
        new_module = deepcopy(module.module)
        is_parallel = True
    else:
        new_module = deepcopy(module)

    dict_layer_out_to_prune = {}
    dict_layer_in_to_prune = {}

    for c in list_channel_to_prune:
        index = c.rfind('.')
        list_layer = c[:index]
        ind_channel = int(c[index + 1:])
        if separator in list_layer:
            list_layer = list_layer.split(separator)
        if not isinstance(list_layer, list):
            list_layer = [list_layer]
        assert isinstance(list_layer, list)
        for layer in list_layer:
            next_layer = compute_next_layer_effnet(layer, module, use_se=use_se)
            if not layer in dict_layer_out_to_prune:
                dict_layer_out_to_prune[layer] = [ind_channel]
            else:
                dict_layer_out_to_prune[layer].append(ind_channel)
            if next_layer is not None:
                if not next_layer in dict_layer_in_to_prune:
                    dict_layer_in_to_prune[next_layer] = [ind_channel]
                else:
                    dict_layer_in_to_prune[next_layer].append(ind_channel)

    new_module = new_module.cuda()
    if is_parallel:
        if use_amp:
            new_module = DDP(new_module)
        elif distributed:
            new_module = torch.nn.parallel.DistributedDataParallel(new_module, device_ids=[local_rank])
        else:
            new_module = torch.nn.DataParallel(new_module)
    for k, v in dict_layer_out_to_prune.items():
        m = extract_layer(new_module, k)
        w = m.weight.data
        b = m.bias
        if b is not None:
            b = b.data
        v = list(set(range(w.shape[0])).difference(v))
        if isinstance(m, Conv2dSame):
            conv = Conv2dSame
        else:
            conv = nn.Conv2d

        if m.groups > 1:
            new_conv = conv(in_channels=len(v), out_channels=len(v),
                            kernel_size=m.kernel_size, bias=b is not None, padding=m.padding, dilation=m.dilation,
                            groups=len(v), stride=m.stride)
            new_conv.weight.data = w[v, :, ...].clone()
        else:
            new_conv = conv(in_channels=m.in_channels, out_channels=len(v),
                            kernel_size=m.kernel_size, bias=b is not None, padding=m.padding, dilation=m.dilation,
                            groups=m.groups, stride=m.stride)
            new_conv.weight.data = w[v, :, ...].clone()
        if b is not None:
            new_conv.bias.data = b[v].clone()
        set_layer(new_module, k, new_conv)
        if '.se.' in k:
            continue
        next_bn_ = compute_next_bn_resnet(k, module)
        next_bn = extract_layer(new_module, next_bn_)
        # new_bn = deepcopy(next_bn)
        # new_bn.num_features = len(v)
        new_bn = nn.BatchNorm2d(num_features=len(v), eps=next_bn.eps, momentum=next_bn.momentum, affine=next_bn.affine,
                                track_running_stats=True)
        new_bn.bias.data = next_bn.bias.data[v].clone()
        new_bn.weight.data = next_bn.weight.data[v].clone()
        new_bn.register_buffer('running_mean', next_bn.running_mean[v].clone())
        new_bn.register_buffer('running_var', next_bn.running_var[v].clone())
        new_bn.register_buffer('num_batches_tracked', next_bn.num_batches_tracked.clone())
        set_layer(new_module, next_bn_, new_bn)

    for k, v in dict_layer_in_to_prune.items():
        m = extract_layer(new_module, k)
        w = m.weight.data
        v = list(set(range(w.shape[1])).difference(v))
        b = m.bias
        if b is not None:
            b = b.data
        if m.groups == 1:
            if isinstance(m, Conv2dSame):
                conv = Conv2dSame
            else:
                conv = nn.Conv2d
            new_conv = conv(in_channels=len(v), out_channels=m.out_channels,
                            kernel_size=m.kernel_size, bias=b is not None, padding=m.padding, dilation=m.dilation,
                            groups=m.groups, stride=m.stride)
            new_conv.weight.data = w[:, v, ...].clone()
            if b is not None:
                new_conv.bias.data = b.clone()
            set_layer(new_module, k, new_conv)

    last_conv_layer = extract_conv_layers(new_module)[-1]
    all_layers = extract_layers(new_module)
    last_layer = all_layers[-1]
    new_module.eval()
    macs_new_module = compute_macs_per_layer(new_module, [3, input_size, input_size])
    module.eval()
    macs_old_module = compute_macs_per_layer(module, [3, input_size, input_size])

    total_mac_old = 0
    total_mac_new = 0
    for k, v in macs_new_module.items():
        total_mac_new += v[1]
    for k, v in macs_old_module.items():
        total_mac_old += v[1]

    logging.info("Actual total mac saved: {0:2.3f}%".format(100 * (total_mac_old - total_mac_new) / total_mac_old))
    logging.info("Previous MAC: {}, new MAC : {}".format(total_mac_old, total_mac_new))
    return new_module


def redesign_module_resnet(module, list_channel_to_prune, use_amp=False, distributed=False, local_rank=0,
                           input_size=224, separator='***'):
    is_parallel = False
    if isinstance(module, DDP) or isinstance(module, nn.DataParallel) or isinstance(module,
                                                                                    nn.parallel.DistributedDataParallel):
        new_module = deepcopy(module.module)
        is_parallel = True
    else:
        new_module = deepcopy(module)

    dict_layer_out_to_prune = {}
    dict_layer_in_to_prune = {}

    for c in list_channel_to_prune:
        index = c.rfind('.')
        list_layer = c[:index]
        ind_channel = int(c[index + 1:])
        if separator in list_layer:
            list_layer = list_layer.split(separator)
        if not isinstance(list_layer, list):
            list_layer = [list_layer]
        assert isinstance(list_layer, list)
        for layer in list_layer:
            next_layer = compute_next_layer_resnet(layer, module)
            if not layer in dict_layer_out_to_prune:
                dict_layer_out_to_prune[layer] = [ind_channel]
            else:
                dict_layer_out_to_prune[layer].append(ind_channel)
            if next_layer is not None:
                if not next_layer in dict_layer_in_to_prune:
                    dict_layer_in_to_prune[next_layer] = [ind_channel]
                else:
                    dict_layer_in_to_prune[next_layer].append(ind_channel)

            next_layer_d = compute_next_downsample(layer, module)
            if next_layer_d is not None:
                if not next_layer_d in dict_layer_in_to_prune:
                    dict_layer_in_to_prune[next_layer_d] = [ind_channel]
                else:
                    dict_layer_in_to_prune[next_layer_d].append(ind_channel)

    new_module = new_module.cuda()
    if is_parallel:
        if use_amp:
            new_module = DDP(new_module)
        elif distributed:
            new_module = torch.nn.parallel.DistributedDataParallel(new_module, device_ids=[local_rank])
        else:
            new_module = torch.nn.DataParallel(new_module)
    for k, v in dict_layer_out_to_prune.items():
        m = extract_layer(new_module, k)
        w = m.weight.data
        b = m.bias
        if b is not None:
            b = b.data
        v = list(set(range(w.shape[0])).difference(v))
        if isinstance(m, Conv2dSame):
            conv = Conv2dSame
        else:
            conv = nn.Conv2d

        if m.groups > 1:
            new_conv = conv(in_channels=len(v), out_channels=len(v),
                            kernel_size=m.kernel_size, bias=b is not None, padding=m.padding, dilation=m.dilation,
                            groups=len(v), stride=m.stride)
            new_conv.weight.data = w[v, :, ...].clone()
        else:
            new_conv = conv(in_channels=m.in_channels, out_channels=len(v),
                            kernel_size=m.kernel_size, bias=b is not None, padding=m.padding, dilation=m.dilation,
                            groups=m.groups, stride=m.stride)
            new_conv.weight.data = w[v, :, ...].clone()
        if b is not None:
            new_conv.bias.data = b[v].clone()
        set_layer(new_module, k, new_conv)
        next_bn_ = compute_next_bn_resnet(k, module)
        next_bn = extract_layer(new_module, next_bn_)
        # new_bn = deepcopy(next_bn)
        # new_bn.num_features = len(v)
        new_bn = nn.BatchNorm2d(num_features=len(v), eps=next_bn.eps, momentum=next_bn.momentum, affine=next_bn.affine,
                                track_running_stats=True)
        new_bn.bias.data = next_bn.bias.data[v].clone()
        new_bn.weight.data = next_bn.weight.data[v].clone()
        new_bn.register_buffer('running_mean', next_bn.running_mean[v].clone())
        new_bn.register_buffer('running_var', next_bn.running_var[v].clone())
        new_bn.register_buffer('num_batches_tracked', next_bn.num_batches_tracked.clone())
        set_layer(new_module, next_bn_, new_bn)

    for k, v in dict_layer_in_to_prune.items():
        m = extract_layer(new_module, k)
        w = m.weight.data
        v = list(set(range(w.shape[1])).difference(v))
        b = m.bias
        if b is not None:
            b = b.data
        if m.groups == 1:
            if isinstance(m, Conv2dSame):
                conv = Conv2dSame
            else:
                conv = nn.Conv2d
            new_conv = conv(in_channels=len(v), out_channels=m.out_channels,
                            kernel_size=m.kernel_size, bias=b is not None, padding=m.padding, dilation=m.dilation,
                            groups=m.groups, stride=m.stride)
            new_conv.weight.data = w[:, v, ...].clone()
            if b is not None:
                new_conv.bias.data = b.clone()
            set_layer(new_module, k, new_conv)

    last_conv_layer = extract_conv_layers(new_module)[-1]
    if '.se.' in last_conv_layer:
        last_conv_layer = extract_conv_layers(new_module)[-2]
    all_layers = extract_layers(new_module)
    last_layer = all_layers[-1]
    num_out_channel = extract_layer(new_module, last_conv_layer).out_channels
    fc_layer = extract_layer(new_module, last_layer)
    assert isinstance(fc_layer, nn.Linear)

    if fc_layer.in_features != num_out_channel:
        index_channel = dict_layer_out_to_prune[last_conv_layer]
        index_channel = list(set(range(fc_layer.in_features)).difference(index_channel))
        new_fc = nn.Linear(in_features=len(index_channel), out_features=fc_layer.out_features,
                           bias=fc_layer.bias is not None)
        new_fc.bias.data = fc_layer.bias.data.clone()
        new_fc.weight.data = fc_layer.weight.data[:, index_channel].clone()
        set_layer(new_module, last_layer, new_fc)

    new_module.eval()
    macs_new_module = compute_macs_per_layer(new_module, [3, input_size, input_size])
    module.eval()
    macs_old_module = compute_macs_per_layer(module, [3, input_size, input_size])

    total_mac_old = 0
    total_mac_new = 0
    for k, v in macs_new_module.items():
        total_mac_new += v[1]
    for k, v in macs_old_module.items():
        total_mac_old += v[1]

    logging.info("Actual total mac saved: {0:2.3f}%".format(100 * (total_mac_old - total_mac_new) / total_mac_old))
    logging.info("Previous MAC: {}, new MAC : {}".format(total_mac_old, total_mac_new))
    return new_module


def prune_network(net, list_channel_to_prune):
    for l in list_channel_to_prune:
        index = l.rfind('.')
        layer = l[:index]
        ind_channel = int(l[index + 1:])
        extract_layer(net, layer).weight[ind_channel, ...] = 0
        if extract_layer(net, layer).bias is not None:
            extract_layer(net, layer).bias[ind_channel] = 0


class FormatterNoInfo(logging.Formatter):
    def __init__(self, fmt='%(levelname)s: %(message)s'):
        logging.Formatter.__init__(self, fmt)

    def format(self, record):
        if record.levelno == logging.INFO:
            return str(record.getMessage())
        return logging.Formatter.format(self, record)


def setup_default_logging(default_level=logging.INFO):
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(FormatterNoInfo())
    logging.root.addHandler(console_handler)
    logging.root.setLevel(default_level)


def last_dim_of_block(mod, n):
    last_conv = None
    last_name = None
    for n2, m2 in mod.named_modules():
        # print(n2)
        if isinstance(m2, nn.Conv2d) and n2 != '':
            last_conv = m2
            last_name = '.'.join(n) + '.' + n2
    if last_name is not None:
        return last_name, last_conv.weight.shape[0]
    return None, None


class build_co_train_model(nn.Module):

    def __init__(self, module1, module2, gamma=0.5, only_last=False, progressive_IKD_factor=False, strict=True):
        super(build_co_train_model, self).__init__()
        self.mod1 = []
        self.mod2 = []
        self.gamma = gamma
        self.last_dim1 = []
        self.last_dim2 = []
        self.index_block = []
        self.mismatch_index = []
        self.M = []
        self.progressive_IKD_factor = progressive_IKD_factor
        self.name = []
        self.block_len = 0
        ind = 0
        for n, m in module1.named_modules():
            n = n.split('.')
            if isinstance(module1, nn.DataParallel) or isinstance(module1, DDP):
                n = n[1:]
            if len(n) == 1 and n[0] != '':
                if n[0] == 'blocks' and not only_last:
                    self.block_len = str(len(module1.blocks._modules) - 1)
                    m_new = deepcopy(m)
                    for n2, m2 in m_new.named_modules():
                        n2 = n2.split('.')
                        if len(n2) == 1 and n2[0] != '':
                            name, s = last_dim_of_block(m2, n2)
                            self.mod1.append(m2)
                            self.name.append(n[0] + '.' + n2[0])
                            if name is not None:
                                self.index_block.append(ind)
                                self.last_dim1.append([name, s])
                            ind += 1
                else:
                    m_new = deepcopy(m)
                    name, s = last_dim_of_block(m_new, n)
                    self.mod1.append(m_new)
                    self.name.append(n[0])
                    if name is not None:
                        self.index_block.append(ind)
                        self.last_dim1.append([name, s])
                    ind += 1

        for n, m in module2.named_modules():
            n = n.split('.')
            if isinstance(module2, nn.DataParallel) or isinstance(module2, DDP):
                n = n[1:]
            if len(n) == 1 and n[0] != '':
                if n[0] == 'blocks' and not only_last:
                    m_new = deepcopy(m)
                    self.add_module('blocks', m_new)
                    for n2, m2 in m_new.named_modules():
                        n2 = n2.split('.')
                        if len(n2) == 1 and n2[0] != '':
                            name, s = last_dim_of_block(m2, n2)
                            self.mod2.append(m2)
                            if name is not None:
                                self.last_dim2.append([name, s])
                else:
                    name, s = last_dim_of_block(m, n)
                    m_new = deepcopy(m)
                    self.mod2.append(m_new)
                    self.add_module(n[0], m_new)
                    if name is not None:
                        self.last_dim2.append([name, s])
        for i, (a, b) in enumerate(zip(self.last_dim1, self.last_dim2)):
            name1 = a[0]
            d1 = a[1]
            name2 = b[0]
            d2 = b[1]
            if strict:
                assert (name1 == name2), f"{name1}, {name2}"
            if d1 != d2:
                logging.info(f'Dimension mismatch for layer {name1} : {d1} and {d2}. Using IKD with linear projection')
                self.mismatch_index.append(self.index_block[i])
                M = nn.Linear(d2, d1, bias=False)
                self.M.append(M)
                self.add_module('_'.join(name1.split('.')) + '_M', M)
        self.no_ikd_fc = False
        if self.last_dim1[-1][1] != self.last_dim2[-1][1]:
            self.no_ikd_fc = True

    def forward(self, x):
        if self.training:
            x1 = x.detach()
        x2 = x
        l = 0
        local_ind = 0
        for i, (m1, m2) in enumerate(zip(self.mod1, self.mod2)):
            if isinstance(m2, nn.Linear):
                s = x2.shape
                o = s[1] * s[2] * s[3]
                if o != m2.in_features:
                    x2 = F.avg_pool2d(x2, x2.size()[3])
                x2 = x2.view(x2.size(0), -1)
            x2 = m2(x2)
            if self.training:
                factor = self.gamma
                if self.progressive_IKD_factor:
                    if 'block' in self.name[i] and self.block_len not in self.name[i]:
                        factor = 0
                    # factor = i ** 2 * self.gamma / (len(self.mod1) ** 2)
                m1.eval()
                if isinstance(m1, nn.Linear):
                    s = x1.shape
                    o = s[1] * s[2] * s[3]
                    if o != m1.in_features:
                        x1 = F.avg_pool2d(x1, x1.size()[3])
                    x1 = x1.view(x1.size(0), -1)
                with torch.no_grad():
                    x1 = m1(x1)
                if i in self.mismatch_index:
                    M = self.M[local_ind]
                    local_ind += 1
                    x1_ = x1.transpose(1, 3)
                    x2_ = x2.transpose(1, 3)
                    l += factor * torch.mean((M(x2_) - x1_) ** 2)
                elif not (isinstance(m1, SelectAdaptivePool2d) and self.no_ikd_fc):
                    l += factor * torch.mean((x2 - x1) ** 2)
        if self.training:
            l /= len(self.mod1)
            return x2, l
        return x2


class build_KD_model(nn.Module):
    def __init__(self, moduleT, moduleS, alpha_KD=0.1, temperature_T=1, temperature_S=1, keep_only_correct=False):
        super(build_KD_model, self).__init__()
        moduleT = moduleT.eval().cuda()
        moduleT = deepcopy(moduleT)
        moduleT.requires_grad = False
        for n, m in moduleT.named_parameters():
            m.requires_grad = False
        self.moduleT = [moduleT]
        self.alpha_KD = alpha_KD
        self.moduleS = [deepcopy(moduleS)]
        for n, m in self.moduleS[0].named_children():
            self.add_module(n, m)
        self.softmax_temperature_T = temperature_T
        self.softmax_temperature_S = temperature_S
        self.keep_only_correct = keep_only_correct

    def forward(self, x, target_gt=None):
        self.moduleT[0] = self.moduleT[0].eval()
        if self.training:
            outS = self.moduleS[0](x)
            loss_KD = 0
            with torch.no_grad():
                outT = self.moduleT[0](x.detach())
                target = F.softmax(outT / self.softmax_temperature_T, dim=-1)
                if self.keep_only_correct:
                    max_value, max_index = torch.max(target, dim=1)
                    mask = (target_gt == max_index).unsqueeze(1).detach().float()
                    target *= mask
            if self.keep_only_correct:
                loss_KD += self.alpha_KD * (F.cross_entropy(outS, max_index, reduction='none') * (1 - mask)).mean()
            input = F.log_softmax(outS / self.softmax_temperature_S, dim=-1)
            loss_KD += self.alpha_KD * F.kl_div(input, target, reduction='batchmean')
            return outS, loss_KD
        else:
            outS = self.moduleS[0](x)
            return outS


def load_module_from_ckpt(parent_module, checkpoint_path, use_ema=False, input_size=224, print_new_size=False):
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict_key = ''
        if isinstance(checkpoint, dict):
            state_dict_key = 'state_dict'
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
            logging.info("Loaded {} from checkpoint '{}'".format(state_dict_key or 'weights', checkpoint_path))
    else:
        logging.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()

    is_module = isinstance(parent_module, DDP) or isinstance(parent_module,
                                                             nn.parallel.DistributedDataParallel) or isinstance(
        parent_module, nn.DataParallel)
    if is_module:
        new_module = deepcopy(parent_module.module)
    else:
        new_module = deepcopy(parent_module)
    for n, m in parent_module.named_modules():
        if is_module:
            n = '.'.join(n.split('.')[1:])
        old_module = extract_layer(parent_module, n)
        if isinstance(old_module, nn.Conv2d) or isinstance(old_module, Conv2dSame):
            if isinstance(old_module, Conv2dSame):
                conv = Conv2dSame
            else:
                conv = nn.Conv2d
            w = state_dict[n + '.weight']
            s = w.shape
            if print_new_size:  # and 'conv2' in n:
                logging.info(f'Convolution: {n}: Old size={m.weight.shape[1::-1]}, new size: {s[1::-1]}')
                # n2=n[0:-5]
                # logging.info(f'({n2}0,{m.weight.shape[1]-s[1]}), ({n2}1,{m.weight.shape[0]-s[0]})')
            in_channels = s[1]
            out_channels = s[0]
            if old_module.groups > 1:
                in_channels = out_channels
                g = in_channels
            else:
                g = 1
            new_conv = conv(in_channels=in_channels, out_channels=out_channels,
                            kernel_size=old_module.kernel_size, bias=old_module.bias is not None,
                            padding=old_module.padding, dilation=old_module.dilation,
                            groups=g, stride=old_module.stride)
            set_layer(new_module, n, new_conv)
        if isinstance(old_module, nn.BatchNorm2d):
            new_bn = nn.BatchNorm2d(num_features=state_dict[n + '.weight'].shape[0], eps=old_module.eps,
                                    momentum=old_module.momentum,
                                    affine=old_module.affine,
                                    track_running_stats=True)
            set_layer(new_module, n, new_bn)
        if isinstance(old_module, nn.Linear):
            w = state_dict[n + '.weight']
            s = w.shape
            if print_new_size:  # and 'conv2' in n:
                logging.info(f'FC: {n}: Old size={m.weight.shape[1::-1]}, new size: {s[1::-1]}')
            in_channels = s[1]
            out_channels = s[0]
            new_fc = nn.Linear(in_features=in_channels, out_features=out_channels, bias=old_module.bias is not None)
            set_layer(new_module, n, new_fc)
    new_module.load_state_dict(state_dict, strict=False)

    new_module.eval()
    parent_module.eval()

    macs_new_module = compute_macs_per_layer(new_module, [3, input_size, input_size])
    macs_old_module = compute_macs_per_layer(parent_module, [3, input_size, input_size])

    total_mac_old = 0
    total_mac_new = 0
    for k, v in macs_new_module.items():
        total_mac_new += v[1]
    for k, v in macs_old_module.items():
        total_mac_old += v[1]

    logging.info("Actual total mac saved: {0:2.3f}%".format(100 * (total_mac_old - total_mac_new) / total_mac_old))
    logging.info("Previous MAC: {}, new MAC : {}".format(total_mac_old, total_mac_new))

    return new_module


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


def measure_cpu_time(module, iterations=10, input_size=224, batch_size=512, fp16=False, only_modules=False):
    class write_time_forward:
        def __init__(self, name):
            self.name = name
            self.time = None
            self.num_call = 0

        def __call__(self, *kwargs):
            nonlocal dict_time
            if self.time is None:
                self.time = time.time()
                self.start = time.time()
            else:
                self.end = time.time()
                time_elapsed = self.end - self.start
                if self.num_call == 0:
                    dict_time[self.name] = time_elapsed
                else:
                    dict_time[self.name] = (self.num_call * dict_time[self.name] + time_elapsed) / (
                            self.num_call + 1)
                self.num_call += 1
                self.time = None
    pre_hook = {}
    post_hook = {}

    module = module.cpu()
    if not (isinstance(input_size, list) or isinstance(input_size, tuple)):
        input_size = [3, input_size, input_size]
    input = torch.rand([batch_size, input_size[0], input_size[1], input_size[2]]).cpu()

    if fp16:
        module = module.half()
        input = input.half()

    dict_time = {}

    module.eval()
    torch.cuda.empty_cache()

    time_total = 0
    with torch.no_grad():
        module(input)

        if not only_modules:
            for _ in tqdm(range(iterations)):
                start = time.time()
                module(input)
                end = time.time()
                time_total += (end - start) / iterations

            print(f"Inference speed is {batch_size / time_total} images/second")

        for n, p in module.named_modules():
            obj = write_time_forward(n)
            pre_hook[n] = p.register_forward_pre_hook(obj)
            post_hook[n] = p.register_forward_hook(obj)
        for _ in tqdm(range(iterations)):
            module(input)

        fixed_time = 0
        for n, m in module.named_modules():
            pre_hook[n].remove()
            post_hook[n].remove()
            if not hasattr(m, 'alpha') and hasattr(m, 'downsample') and m.downsample is not None:
                fixed_time += dict_time[n]

    module.train()

    return time_total, dict_time, fixed_time


def measure_time(module, iterations=10, input_size=224, batch_size=512, fp16=False, only_modules=False):
    class write_time_forward:
        def __init__(self, name):
            self.name = name
            self.time = None
            self.num_call = 0
            self.start = torch.cuda.Event(enable_timing=True)
            self.end = torch.cuda.Event(enable_timing=True)

        def __call__(self, *kwargs):
            with torch.cuda.device(kwargs[1][0].device):
                nonlocal dict_time
                if self.time is None:
                    self.time = time.time()
                    self.start.record()
                else:
                    self.end.record()
                    torch.cuda.synchronize(device=kwargs[1][0].device)
                    time_elapsed = self.start.elapsed_time(self.end) / 1000.0
                    if self.num_call == 0:
                        dict_time[self.name] = time_elapsed
                    else:
                        dict_time[self.name] = (self.num_call * dict_time[self.name] + time_elapsed) / (
                                self.num_call + 1)
                    self.num_call += 1
                    self.time = None

    pre_hook = {}
    post_hook = {}

    module = module.cuda()
    if not (isinstance(input_size, list) or isinstance(input_size, tuple)):
        input_size = [3, input_size, input_size]
    input = torch.rand([batch_size, input_size[0], input_size[1], input_size[2]]).cuda()

    if fp16:
        module = module.half()
        input = input.half()

    dict_time = {}

    module.eval()
    torch.cuda.empty_cache()
    with torch.no_grad():
        module(input)

    time_total = 0
    if not only_modules:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(iterations):
            with torch.no_grad():
                start.record()
                module(input)
                end.record()
                torch.cuda.synchronize()
                time_total += start.elapsed_time(end) / (1000.0 * iterations)

        print(f"Inference speed is {batch_size / time_total} images/second")

    for n, p in module.named_modules():
        obj = write_time_forward(n)
        pre_hook[n] = p.register_forward_pre_hook(obj)
        post_hook[n] = p.register_forward_hook(obj)
    for _ in range(iterations):
        with torch.no_grad():
            module(input)

    fixed_time = 0
    for n, m in module.named_modules():
        pre_hook[n].remove()
        post_hook[n].remove()
        if not hasattr(m, 'alpha') and hasattr(m, 'downsample') and m.downsample is not None:
            fixed_time += dict_time[n]

    if fp16:
        module = module.float()

    module.train()

    return time_total, dict_time, fixed_time
