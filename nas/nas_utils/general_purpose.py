# from apex import amp
import gc
import logging
import re
import time
from contextlib import suppress

import numpy as np

from external.distributed_manager import DistributedManager, master_only
from timm.models.efficientnet_blocks import InvertedResidual
from timm.models.layers import hard_sigmoid
from timm.models.layers.activations_me import *
from timm.models.mobilenasnet import InvertedResidualElastic, MobileNasNet, SinkPoint, \
    generate_latency_dict

EPSILON = 1e-8

EXP_RATIO = [3, 4, 6]
EXP_RATIO_EXTENDED = [2, 2.5, 3, 4, 6]
DW_K_SIZE = [3, 5]
SE_RATIO = [0, 0.25]


def extract_module_name(model, module):
    model_ = model
    if hasattr(model_, 'module'):
        model_ = model_.module
    name = [name for name, _module in model_.named_modules() if _module == module][0]
    return name


def set_resolution_hook(model):
    resolution_dict = {}

    def callback_hook_resolution(resolution_dict):
        def hook(module, input, output):
            name = extract_module_name(model, module)
            res = input[0].shape[-1]
            resolution_dict[name] = res

        return hook

    list_hook = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            list_hook.append(m.register_forward_hook(callback_hook_resolution(resolution_dict)))
    return resolution_dict, list_hook


def compute_resolution(model):
    model.eval()
    model.cpu()
    resolution_dict, list_hook = set_resolution_hook(model)
    with torch.no_grad():
        model(torch.rand(10, 3, 224, 224))
    for h in list_hook:
        h.remove()
    model.cuda()
    return resolution_dict


class AlphaWrapper(nn.Module):
    def __init__(self, module_list, init_alpha=None, force_gs=False, only_max=False, force_sm=False, retain_grad=True,
                 fixed_alpha_max=False, hard_backprop_gs=False):
        super(AlphaWrapper, self).__init__()
        self.force_gs = force_gs
        self.fixed_alpha_max = fixed_alpha_max
        self.hard_backprop_gs = hard_backprop_gs
        assert not (fixed_alpha_max and init_alpha is None)
        if init_alpha is None:
            init_alpha = [1.0] * (len(module_list) + 1)
        if fixed_alpha_max:
            index = torch.argmax(torch.tensor(init_alpha))
            module_list = [module_list[index] if index <= len(module_list) - 1 else nn.Identity()]
            self.alpha = 1.0
        else:
            self.register_parameter(name='alpha',
                                    param=torch.nn.parameter.Parameter(torch.tensor(init_alpha), requires_grad=False))
            assert len(module_list) + 1 == len(init_alpha)
        self._only_max = only_max
        self._attention = None
        self.force_sm = force_sm
        self.retain_grad = retain_grad
        self.sub_module = torch.nn.ModuleList(module_list)

    @property
    def attention(self):
        if self._attention is None:
            self._attention = torch.softmax(self.alpha, dim=0)
        return self._attention

    @property
    def only_max(self):
        return self._only_max

    @only_max.setter
    def only_max(self, val):
        if val is True:
            self._only_max = True
            index = torch.argmax(self.alpha)
            self.alpha.data = torch.zeros_like(self.alpha)
            self.alpha[index] = 1.0
            self.alpha.requires_grad = False
            self._attention = self.alpha.data
        else:
            self._only_max = False

    def forward(self, input):
        if self.fixed_alpha_max:
            return self.sub_module[0](input)
        if (self.training or self.force_gs) and not self._only_max and not self.force_sm:
            self._attention = nn.functional.gumbel_softmax(self.alpha, hard=True, dim=0, tau=1, eps=1e-10)
            if DistributedManager.distributed and self.hard_backprop_gs:
                group = DistributedManager.grp
                torch.distributed.broadcast(self._attention, 0, group=group)
            if self.retain_grad:
                self._attention.retain_grad()
        elif not self._only_max:
            self._attention = torch.softmax(self.alpha, dim=0)
            if self.retain_grad:
                self._attention.retain_grad()
        else:
            index = torch.argmax(self.alpha)
            self.alpha.data = torch.zeros_like(self.alpha)
            self.alpha[index] = 1.0
            self.alpha.requires_grad = False
            self._attention = self.alpha.data
        out = 0
        for u, sub_module in zip(self._attention, self.sub_module):
            if (self.training and not self._only_max and not self.hard_backprop_gs) or u > 0:
                out += u * sub_module(input)

        if (self.training and not self._only_max) or self._attention[-1] > 0:
            out += self._attention[-1] * input

        return out


class CustomError(Exception):
    pass



class extend_block(nn.Module):
    def __init__(self, b, e_r, index_k, is_se):
        super(extend_block, self).__init__()
        self.sub_block = [b]
        if e_r == EXP_RATIO_EXTENDED[-1] and index_k == len(DW_K_SIZE) - 1 and is_se == 1:
            for n, m in b.named_modules():
                if n != '' and '.' not in n:
                    self.add_module(n, m)
        self.e_r = e_r
        self.index_k = index_k
        self.is_se = is_se

    def forward(self, x):
        return self.sub_block[0](x, self.e_r, self.index_k, self.is_se)

    def train(self, mode=True):
        super(extend_block, self).train(mode)
        self.sub_block[0].train(mode)



STAGE_BLOCK_DELIMITER = '-'
BLOCK_ALPHA_CONFIG_DELIMITER = '|'


def er_k_se_str_by_attention_index(elastic_ir, alpha_config_ind):
    return 'er_{}_k_{}_se_{}'.format(*elastic_ir.er_k_se_by_attention_index(alpha_config_ind))


def stage_block_name(stage_ind, block_ind):
    return 'stage_{}{}block_{}'.format(stage_ind, STAGE_BLOCK_DELIMITER, block_ind)


def stage_block_alpha_config_name(stage_id, block_id, block, alpha_config_ind):
    stage_block_str = stage_block_name(stage_id, block_id)
    alpha_config_str = er_k_se_str_by_attention_index(block, alpha_config_ind)
    return BLOCK_ALPHA_CONFIG_DELIMITER.join([stage_block_str, alpha_config_str])


def get_stage_block_from_name(name, splitted=True):
    if splitted:
        return tuple(name.split(BLOCK_ALPHA_CONFIG_DELIMITER)[0].split(STAGE_BLOCK_DELIMITER))
    else:
        return name.split(BLOCK_ALPHA_CONFIG_DELIMITER)[0]


def get_block_alpha_config_from_name(name):
    return name.split(STAGE_BLOCK_DELIMITER)[-1]


def get_config_from_name(name):
    return name.split(BLOCK_ALPHA_CONFIG_DELIMITER)[-1]


LATENCY_FILENAME = 'lut.pkl'


def _extract_alpha_beta_and_associated_modules_elastic(model, file_name=LATENCY_FILENAME,
                                                       batch_size=1, repeat_measure=200, target_device='cpu'):
    assert isinstance(model, MobileNasNet)
    list_betas = []

    sink_points = [(s, stage) for s, stage in enumerate(model.sink_points) if isinstance(stage, SinkPoint)]
    assert len(sink_points) == len(model.beta_attention) and len(sink_points) == len(model.beta)

    for (s, stage), beta_attention, beta in zip(sink_points, model.beta_attention, model.beta):
        block_list = []
        alpha_entries = []
        for b, block in enumerate(stage):
            if not isinstance(block, InvertedResidualElastic):
                continue

            block_list.append(stage_block_name(s, b))
            list_submodule = [stage_block_alpha_config_name(s, b, block, i) for i in range(len(block))]
            alpha_entry = dict(submodules=list_submodule, attention=block._attention, module=block,
                               alpha=block.alpha, _alpha=block._alpha, latency=None)
            alpha_entries.append(alpha_entry)

        entry = dict(submodules=block_list, attention=beta_attention, module=stage,
                     beta=beta, alpha_entries=alpha_entries)

        list_betas.append(entry)
        list_betas += alpha_entries

    dict_latency = compute_latency(model, list_betas, file_name=file_name, batch_size=batch_size,
                                   repeat_measure=repeat_measure, target_device=target_device)
    fixed_latency = dict_latency['general']
    for entry in list_betas:
        if isinstance(entry, dict) and 'alpha_entries' in entry.keys():
            alpha_entries = entry['alpha_entries']
            for alpha_entry in alpha_entries:
                alpha_entry['latency'] = torch.tensor([dict_latency[s][1] for s in alpha_entry['submodules']])

    return list_betas, fixed_latency


def compute_latency(model, struct, file_name=LATENCY_FILENAME, batch_size=1, repeat_measure=200, target_device='cpu'):
    desc_blocks = extract_resolution_stride_dict(model, struct)

    dict_time = generate_latency_dict(model, file_name=file_name, batch_size=batch_size,
                                      iterations=repeat_measure, target=target_device, raw=True)
    for k, v in desc_blocks.items():
        desc_blocks[k] = [v, dict_time[v]]

    desc_blocks['general'] = dict_time['general']
    return desc_blocks


def extract_structure_param_list(model,
                                 file_name=LATENCY_FILENAME, batch_size=1, repeat_measure=200, target_device='cpu'):
    return _extract_alpha_beta_and_associated_modules_elastic(model, file_name=file_name,
                                                                  batch_size=batch_size,
                                                                  repeat_measure=repeat_measure,
                                                                  target_device=target_device)


def extract_resolution_stride_dict(model, struct):
    desc_blocks = {}
    resolution = compute_resolution(model)
    for e in struct:
        if 'submodules' in e and not 'alpha_entries' in e:
            alpha_lst = e['submodules']
            for submodule in alpha_lst:
                stride = e['module'].conv_dw[0].stride[0]
                in_ch = e['module'].conv_pw.in_channels
                out_ch = e['module'].conv_pwl.out_channels if isinstance(e['module'].conv_pwl, nn.Conv2d) else \
                    e['module'].conv_pwl[0].out_channels
                res = resolution[extract_module_name(model, e['module'].conv_pw)]
                use_relu = int(isinstance(e['module'].act1, nn.ReLU))
                desc_blocks[
                    submodule] = f'in_channels_{in_ch}_out_channels_{out_ch}_stride_{stride}_resolution_{res}_er_' + \
                                 submodule.split('er_')[1] + f'_act_{use_relu}'

    return desc_blocks


def measure_time_inverted_bottleneck(desc, batch_size=1, num_iter=200):
    logging.info(f"Measuring inference time for submodule with parameters: {desc}")
    m = re.split(
        'in_channels_(\d+)_out_channels_(\d+)_stride_(\d)_resolution_(\d+)_er_(\d.?\d?\d?)_k_(\d)_se_(\d.?\d?\d?)_act_(\d)',
        desc)
    in_channels = int(m[1])
    out_channels = int(m[2])
    stride = int(m[3])
    resolution = int(m[4])
    er = float(m[5])
    k = int(m[6])
    se = float(m[7])
    use_relu = int(m[8])
    if use_relu:
        act = nn.ReLU
    else:
        act = HardSwishMe
    se_kwargs = dict(act_layer=nn.ReLU, gate_fn=hard_sigmoid, reduce_mid=True, divisor=8)
    module = InvertedResidual(in_chs=in_channels, out_chs=out_channels, dw_kernel_size=k, stride=stride, act_layer=act,
                              exp_ratio=er, se_ratio=se, se_kwargs=se_kwargs)
    input = torch.rand(batch_size, in_channels, resolution, resolution)

    with torch.no_grad():
        res = module(input)
    start = time.time()
    with torch.no_grad():
        for i in range(num_iter):
            res = module(input)
    end = time.time()
    del module, input, res
    gc.collect()
    return (end - start) / num_iter


def expected_latency(list_alphas):
    latency = 0
    for entry in list_alphas:
        if 'alpha' in entry:
            continue

        aggergated_stage = 0
        beta_attentions = torch.softmax(entry['beta'], dim=0)  # .cpu().numpy()
        fixed = len(entry['alpha_entries']) - len(beta_attentions)
        for alpha_entry in entry['alpha_entries'][:fixed]:
            alpha_attentions = torch.softmax(alpha_entry['module'].alpha, dim=0)  # .cpu().numpy()
            aggergated_stage = aggergated_stage + (alpha_attentions * alpha_entry['latency']).sum()

        for beta_attention, alpha_entry in zip(beta_attentions, entry['alpha_entries'][fixed:]):
            alpha_attentions = torch.softmax(alpha_entry['module'].alpha, dim=0)
            aggergated_stage = aggergated_stage + (alpha_attentions * alpha_entry['latency']).sum()
            latency = latency + beta_attention * aggergated_stage

    return latency


def target_time_loss(list_alphas, t_max):
    t = expected_latency(list_alphas)
    loss = torch.relu(t / t_max - 1)
    return loss


def inference_time_loss(list_alphas):
    return expected_latency(list_alphas)

def unfreeze_all(model, optim_alpha):
    # Enable weights gradient, BN statistics and alpha gradients
    model.train()
    model.requires_grad_(True)
    optim_alpha.requires_grad_(True)

def freeze_all(model, optim_alpha):
    # Enable weights gradient, BN statistics and alpha gradients
    model.train()
    model.requires_grad_(False)
    optim_alpha.requires_grad_(False)


def freeze_alphas_unfreeze_weights(model, optim_alpha):
    # Enable weights gradients and BN statistics and disable alpha gradients
    model.train()
    if hasattr(model, 'module'):
        model = model.module
    model.set_require_grad(True)


def freeze_weights_unfreeze_alphas(model, optim_alpha):
    # Disable weights gradients and BN statistics and enable alpha gradients
    model.eval()
    if hasattr(model, 'module'):
        model = model.module
    model.set_require_grad(False)


class OptimLike(object):
    pass


def alpha_bilevel_backward(model, loader, loss_fn, args, list_alphas, optim_alpha, amp_autocast=suppress,
                           loss_scaler=None):
    # Disable weights gradients and BN statistics and enable alpha gradients
    freeze_weights_unfreeze_alphas(model, optim_alpha)
    second_order = hasattr(optim_alpha, 'is_second_order') and optim_alpha.is_second_order

    # Zero out aggregated alpha gradients
    optim_alpha.zero_grad()
    model.zero_grad()

    # Get next batch from the secondary loader (e.g. validation set loader)
    input, target = next(loader)
    if not args.prefetcher:
        input, target = input.cuda(), target.cuda()

    # Compute the output
    with amp_autocast():
        output = model(input)
        if isinstance(output, (tuple, list)):
            output = output[0]

        # Compute the loss
        loss = loss_fn(output, target)


    # Backprop
    if loss_scaler is not None:
        lst_attention = []
        for n, m in model.named_modules():
            if hasattr(m, '_attention'):
                lst_attention.append(m._attention)
            if hasattr(m, 'beta_attention'):
                for p in m.beta_attention:
                    lst_attention.append(p)

        optim_attention = OptimLike()
        optim_attention.param_groups = [{'params': lst_attention}]

        loss_scaler(
            loss, optim_alpha, clip_grad=args.clip_grad, parameters=model.parameters(), create_graph=second_order,
            unscale=True, add_opt=optim_attention, step=False)
    else:
        loss.backward(create_graph=second_order)

    # Enable weights gradients and BN statistics and disable alpha gradients
    freeze_alphas_unfreeze_weights(model, optim_alpha)

    return input


@master_only
def update_alpha_beta_tensorboard(epoch, list_alphas, writer, latency=None):
    if writer is None or list_alphas is None:
        return

    entropy = {}
    for e, entry in enumerate(list_alphas):
        key = 'alpha' if 'alpha' in entry else 'beta'
        stage, block = get_stage_block_from_name(entry['submodules'][0])
        title = '_'.join([key, stage])
        logits = entry['module'].alpha if key == 'alpha' else entry[key]
        probs = torch.softmax(logits, dim=0).detach().cpu().numpy()

        submodules_captions = [get_block_alpha_config_from_name(submodule_name)
                               if key == 'alpha' else 'block_{}'.format(i)
                               for i, submodule_name in enumerate(entry['submodules'])]
        plots = {submodule_caption: prob for submodule_caption, prob in zip(submodules_captions, probs)}
        writer.add_scalars(title, plots, epoch)

        if key == 'alpha':
            alpha_per_block = get_stage_block_from_name(entry['submodules'][0], splitted=False)
            submodules_captions = [get_config_from_name(submodule_name) for submodule_name in entry['submodules']]
            plots = {submodule_caption: prob for submodule_caption, prob in zip(submodules_captions, probs)}
            writer.add_scalars(alpha_per_block, plots, epoch)

        grads = entry['module'].attention_grad if key == 'alpha' else entry['module'].beta_attention_grad
        if grads is None:
            grads = torch.zeros_like(logits)
        else:
            grads = grads.detach().clone()
            grads = torch.mean(grads, dim=1) if len(probs.shape) < len(grads.shape) else grads
            grads = grads.cpu().numpy()

        plots = {STAGE_BLOCK_DELIMITER.join(['grad', submodule_caption]): grad
                 for submodule_caption, grad in zip(submodules_captions, grads)}

        writer.add_scalars(STAGE_BLOCK_DELIMITER.join(['grads', title]), plots, epoch)

        grads = entry['module'].alpha_grad if key == 'alpha' else entry[key].grad
        if grads is None:
            grads = torch.zeros_like(logits)
        else:
            grads = grads.detach().clone()
            grads = torch.mean(grads, dim=1) if len(probs.shape) < len(grads.shape) else grads
            grads = grads.cpu().numpy()

        plots = {STAGE_BLOCK_DELIMITER.join(['logits_grad', submodule_caption]): grad
                 for submodule_caption, grad in zip(submodules_captions, grads)}

        writer.add_scalars(STAGE_BLOCK_DELIMITER.join(['logits_grads', title]), plots, epoch)

        record_entropy(entropy, key, stage, block, probs)

    mean_entropies(entropy, writer, epoch)

    if latency is not None:
        writer.add_scalars('latency', latency, epoch)


def record_entropy(entropy, key, stage, block, probs):
    probs[probs == 0] = 1
    log_probs = np.log(probs)
    # log_probs[np.isinf(log_probs)] = 0
    normalized_entropy = -np.dot(probs, log_probs) / np.log(len(probs))
    if stage not in entropy:
        entropy[stage] = {}

    entropy[stage][key if key == 'beta' else block] = normalized_entropy


def mean_entropies(entropy, writer=None, epoch=None):
    total_alpha_entropy = []
    total_beta_entropy = []
    for stage, entropies in entropy.items():
        entropies['mean_alphas'] = np.mean([block_entropy for k, block_entropy in entropies.items() if k != 'beta'])
        total_alpha_entropy.append(entropies['mean_alphas'])
        total_beta_entropy.append(entropies['beta'])
        if writer is None or epoch is None: continue

        writer.add_scalars(STAGE_BLOCK_DELIMITER.join(['entropy', stage]), entropies, epoch)

    total_alpha_entropy = np.mean(total_alpha_entropy)
    total_beta_entropy = np.mean(total_beta_entropy)
    total_entropy = np.mean([total_alpha_entropy, total_beta_entropy])

    if writer is not None and epoch is not None:
        plots = {'alpha': total_alpha_entropy, 'beta': total_beta_entropy, 'total': total_entropy}
        writer.add_scalars('total_entropy', plots, epoch)

    return total_alpha_entropy, total_beta_entropy, total_entropy


