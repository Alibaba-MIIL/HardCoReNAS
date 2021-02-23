import gc
import itertools
import logging
import os
import pickle
import re
import time
from copy import deepcopy
from functools import partial

import numpy as np

from external.distributed_manager import DistributedManager
from external.utils_pruning import extract_layer, extract_conv_layers, set_layer, measure_cpu_time
from external.utils_pruning import measure_time as measure_gpu_time
from timm.data import resolve_data_config, create_loader
from timm.models.layers.activations_me import *
from .efficientnet_blocks import resolve_act_layer, InvertedResidual, make_divisible, resolve_se_args, SqueezeExcite
from .efficientnet_builder import decode_arch_def, resolve_bn_args
from .layers import create_conv2d, drop_path
from .layers import hard_sigmoid
from .layers.activations import sigmoid
from .mobilenetv3 import MobileNetV3

try:
    from apex import amp
except ImportError:
    pass

try:
    import onnx
    import onnxruntime as ort
except ImportError:
    logging.warning("ONNX and onnxruntime not found")
    pass

DEPTH_UNIT = [2, 3, 4]
EXP_RATIO = [3, 4, 6]
EXP_RATIO_EXTENDED = [2, 2.5, 3, 4, 6]
DW_K_SIZE = [3, 5]
SE_RATIO = [0, 0.25]

from .registry import register_model

__all__ = ['mobilenasnet']


class InvertedResidualElastic(nn.Module):
    """ Inverted residual block w/ optional SE and CondConv routing"""

    def __init__(self, in_chs, out_chs, dw_kernel_size=DW_K_SIZE,
                 stride=1, dilation=1, pad_type='', act_layer=nn.ReLU, noskip=False,
                 exp_ratio=EXP_RATIO_EXTENDED, exp_kernel_size=1, pw_kernel_size=1,
                 se_ratio=SE_RATIO, se_kwargs=None, norm_layer=nn.BatchNorm2d, norm_kwargs=None,
                 conv_kwargs=None, drop_path_rate=0., init_alpha=None, hard_backprop=False,
                 retain_grad=True, use_only_alpha=False, use_dedicated_pwl_se=False, force_sync_gpu=False,
                 use_privatized_bn=True, multipath_sampling=False, use_softmax=False, search_mode=True):
        super(InvertedResidualElastic, self).__init__()
        norm_kwargs = norm_kwargs or {}
        conv_kwargs = conv_kwargs or {}
        if init_alpha is None:
            init_alpha = [1.0] * (len(exp_ratio) * len(dw_kernel_size) * len(se_ratio))
            init_alpha = torch.tensor(init_alpha) / len(init_alpha)
        self.register_parameter(name='_alpha',
                                param=torch.nn.parameter.Parameter(init_alpha, requires_grad=False))
        self.search_mode = search_mode
        self.use_privatized_bn = use_privatized_bn
        self._attention = init_alpha.clone().detach()
        self._hard_backprop = hard_backprop
        self.retain_grad = retain_grad
        self.use_only_alpha = use_only_alpha
        self.use_dedicated_pwl_se = use_dedicated_pwl_se
        self.force_sync_gpu = force_sync_gpu
        self.multipath_sampling = multipath_sampling
        self.force_se = False
        self._aggregared_grad = None
        self.use_softmax = use_softmax
        assert isinstance(dw_kernel_size, list)
        assert isinstance(exp_ratio, list)
        assert isinstance(se_ratio, list)
        assert len(se_ratio) == 2
        assert se_ratio[0] == 0
        self.has_residual = (in_chs == out_chs and stride == 1) and not noskip
        self.drop_path_rate = drop_path_rate
        self.exp_kernel_size = exp_kernel_size
        self.dw_kernel_size = dw_kernel_size
        self.se_ratio = se_ratio
        self.exp_ratio = exp_ratio
        self._attention_er = torch.zeros(len(self.exp_ratio), requires_grad=False)
        self._attention_se = torch.zeros(len(self.se_ratio), requires_grad=False)
        self._attention_k = torch.zeros(len(self.dw_kernel_size), requires_grad=False)
        self._attention_grad = None
        self._temperature = 1
        exp_ratio = exp_ratio[-1]
        mid_chs = make_divisible(in_chs * exp_ratio)
        list_mid_chs = [make_divisible(in_chs * e) for e in self.exp_ratio]
        # Point-wise expansion
        factors = [float(in_chs * r) / mid_chs for r in self.exp_ratio if r < max(self.exp_ratio)]
        object.__setattr__(self, 'shrinker', WidthShrinker(self._attention_er, factors, mid_chs))

        self.conv_pw = create_conv2d(in_chs, mid_chs, exp_kernel_size, padding=pad_type, **conv_kwargs)
        self.bn1 = norm_layer(mid_chs, **norm_kwargs)
        self.act1 = act_layer(inplace=True)

        # Depth-wise convolution

        conv_dw_lst = []
        bn2_lst = []
        self.conv_dw_bn_lst = []
        for k in dw_kernel_size:
            conv_dw = create_conv2d(
                mid_chs, mid_chs, k, stride=stride, dilation=dilation,
                padding=pad_type, depthwise=True, **conv_kwargs)
            conv_dw_lst.append(conv_dw)
            if self.use_privatized_bn or k == dw_kernel_size[0]:
                bn2 = norm_layer(mid_chs, **norm_kwargs)
                bn2_lst.append(bn2)

            self.conv_dw_bn_lst.append(nn.Sequential(conv_dw, bn2))

        self.conv_dw = nn.ModuleList(conv_dw_lst)
        self.bn2 = nn.ModuleList(bn2_lst)
        object.__setattr__(self, 'conv_dw_bn', AttentionWrapper(self._attention_k, self.conv_dw_bn_lst))

        self.act2 = act_layer(inplace=True)

        # Squeeze-and-excitation

        se_kwargs = resolve_se_args(se_kwargs, in_chs, act_layer)
        # self.se = nn.ModuleList(
        #     [SqueezeExcite(mid_chs, se_ratio=se_r, **se_kwargs) if se_r != 0 else nn.Identity() for se_r in
        #      self.se_ratio])
        self.se = nn.ModuleList(
            [MaskedSqueezeExcite(mid_chs, se_ratio=se_r, list_in_chs=list_mid_chs, attention=self._attention_er,
                                 **se_kwargs) if se_r != 0 else nn.Identity() for se_r in
             self.se_ratio])
        object.__setattr__(self, 'se_op', AttentionWrapper(self._attention_se, self.se))

        # Point-wise linear projection
        if use_dedicated_pwl_se:
            self.conv_pwl_lst = []
            for i in range(len(self.se)):
                conv_pwl = create_conv2d(mid_chs, out_chs, pw_kernel_size, padding=pad_type, **conv_kwargs)
                self.conv_pwl_lst.append(conv_pwl)
            self.conv_pwl = nn.ModuleList(self.conv_pwl_lst)
            object.__setattr__(self, 'conv_pwl_se', AttentionWrapper(self._attention_se, self.conv_pwl_lst))
        else:
            self.conv_pwl = create_conv2d(mid_chs, out_chs, pw_kernel_size, padding=pad_type, **conv_kwargs)
        # self.bn3 = norm_layer(out_chs, **norm_kwargs)
        bn3_lst = []
        # Here, we use a trick from Slimmable network, every widths has its own batchnorm

        # num_bn3 = len(self.exp_ratio)
        if self.use_privatized_bn:
            num_bn3 = len(self.exp_ratio) * len(self.se_ratio)
            for i in range(num_bn3):
                bn3 = norm_layer(out_chs, **norm_kwargs)
                bn3_lst.append(bn3)
        else:
            bn3 = norm_layer(out_chs, **norm_kwargs)
            bn3_lst.append(bn3)

        self.bn3 = nn.ModuleList(bn3_lst)

    def feature_info(self, location):
        if location == 'expansion':
            info = dict(module='act1', hook_type='forward', num_chs=self.conv_pw.in_channels)
        elif location == 'depthwise':  # after SE
            info = dict(module='conv_pwl', hook_type='forward_pre', num_chs=self.conv_pwl.in_channels)
        else:  # location == 'bottleneck'
            info = dict(module='', hook_type='', num_chs=self.conv_pwl.out_channels)
        return info

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, val):
        self._temperature = val

    @property
    def alpha(self):
        if self.force_se:
            return self._alpha[1::2]
        return self._alpha

    @alpha.setter
    def alpha(self, val):
        if self.force_se:
            self._alpha[1::2] = val
            self._alpha[0::2] = -float('inf')
        else:
            self._alpha = val

    @property
    def alpha_grad(self):
        if self._alpha.grad is None:
            return None
        if self.force_se:
            return self._alpha.grad[1::2]
        return self._alpha.grad

    @property
    def attention(self, temperature=1):
        if self._attention is None:
            self._attention = nn.functional.softmax(self._alpha / temperature, dim=0)
        if self.force_se:
            return self._attention[1::2]
        return self._attention

    @attention.setter
    def attention(self, attention, temperature=1):
        if self._attention is None:
            self._attention = nn.functional.softmax(self._alpha / temperature, dim=0)

        if self.force_se:
            self._attention[1::2] = attention
        else:
            self._attention.data = attention

    @property
    def attention_grad(self):
        if self._attention_grad is not None:
            return self._attention_grad
        if self._attention.grad is None:
            return None
        if self.force_se:
            return self._attention.grad[1::2]
        return self._attention.grad

    @attention_grad.setter
    def attention_grad(self, grad):
        self._attention_grad = grad

    def forward(self, x):
        residual = x

        self.initialize_attentions(x)
        if not self.search_mode:
            self.distribute_attentions(x, use_only_alpha=self.use_only_alpha)
            self.update_attention_wrappers()
            out = self._forward(x)
        else:
            gs_attention = self._attention
            if self.hard_backprop and not len(self.attention[0].shape) > 0:
                indices = [np.argmax(self._attention.detach().cpu().numpy())]
            else:
                indices = range(len(gs_attention))

            out = 0
            for i in indices:
                self._attention = torch.zeros(self._alpha.shape)
                self._attention[i] = 1
                self.distribute_attentions(x, use_only_alpha=True)
                self.update_attention_wrappers(hard_backprob=True)

                a = gs_attention[i]
                if not self.hard_backprop or torch.any(a > 0):
                    o = self._forward(x)
                    a = a.view(-1, 1, 1, 1) if len(a.shape) > 0 else a
                    out += a * o

            self._attention = gs_attention

        # Residual
        if self.has_residual:
            if self.drop_path_rate > 0.:
                out = drop_path(out, self.drop_path_rate, self.training)
            out += residual

        return out

    def _forward(self, x):
        # Point-wise expansion
        x = self.conv_pw(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv_dw_bn(x)
        x = self.shrinker(x)
        x = self.act2(x)
        # Squeeze-and-excitation
        x = self.se_op(x)
        # Point-wise linear projection
        if self.use_dedicated_pwl_se:
            x = self.conv_pwl_se(x)
        else:
            x = self.conv_pwl(x)

        if self.multipath_sampling and self.use_privatized_bn and not self.search_mode:
            o = 0
            for i, er in enumerate(self._attention_er.detach()):
                for j, se in enumerate(self._attention_se.detach()):
                    o += er.view(-1, 1, 1, 1) * se.view(-1, 1, 1, 1) * self.bn3[i * len(self.se_ratio) + j](x)
            x = o
        else:
            if self.use_privatized_bn:
                selected_mask1 = np.argmax(self._attention_er.detach().cpu().numpy())
                selected_mask2 = np.argmax(self._attention_se.detach().cpu().numpy())
                selected_mask = selected_mask1 * len(self.se_ratio) + selected_mask2
            else:
                selected_mask = 0
            # selected_mask = np.argmax(self._attention_er.detach().cpu().numpy())
            x = self.bn3[selected_mask](x)

        return x

    def distribute_attentions(self, x, use_only_alpha):
        if self.search_mode:
            batch_size = int(x.shape[0])
            if self.multipath_sampling and not use_only_alpha:
                self._attention_er = torch.zeros(len(self.exp_ratio), batch_size, dtype=x.dtype, device=x.device)
                self._attention_se = torch.zeros(len(self.se_ratio), batch_size, dtype=x.dtype, device=x.device)
                self._attention_k = torch.zeros(len(self.dw_kernel_size), batch_size, dtype=x.dtype, device=x.device)
            else:
                self._attention_er = torch.zeros(len(self.exp_ratio), dtype=x.dtype, device=x.device)
                self._attention_se = torch.zeros(len(self.se_ratio), dtype=x.dtype, device=x.device)
                self._attention_k = torch.zeros(len(self.dw_kernel_size), dtype=x.dtype, device=x.device)

        index_alpha = 0
        for e in range(len(self.exp_ratio)):
            for k in range(len(self.dw_kernel_size)):
                for se in range(len(self.se_ratio)):
                    if self.multipath_sampling and not use_only_alpha:
                        self._attention_er[e, :] += self._attention[index_alpha, :]
                        self._attention_k[k, :] += self._attention[index_alpha, :]
                        self._attention_se[se, :] += self._attention[index_alpha, :]
                    else:
                        self._attention_er[e] += self._attention[index_alpha]
                        self._attention_k[k] += self._attention[index_alpha]
                        self._attention_se[se] += self._attention[index_alpha]
                    index_alpha += 1

        self.conv_dw_bn.attention = self._attention_k
        self.se_op.attention = self._attention_se
        if self.use_dedicated_pwl_se:
            self.conv_pwl_se.attention = self._attention_se
        self.shrinker.attention = self._attention_er
        for m in self.se:
            if isinstance(m, MaskedSqueezeExcite):
                m.attention = self._attention_er
                m.shrinker.attention = self._attention_er

    def initialize_attentions(self, x):
        batch_size = int(x.shape[0])
        if not self.use_only_alpha:
            if self.force_se:
                v = torch.zeros_like(self._alpha.data, requires_grad=False)
                v[0::2] = -float('inf')
                self._alpha.data += v

            if self.multipath_sampling:
                if not self.use_softmax:
                    self._attention = nn.functional.gumbel_softmax(
                        self._alpha.repeat(batch_size).reshape(batch_size, len(self._alpha)).transpose(0, 1),
                        hard=True, dim=0, tau=self._temperature, eps=1e-10)
                else:
                    self._attention = nn.functional.softmax(
                        self._alpha.repeat(batch_size).reshape(batch_size, len(self._alpha)).transpose(0, 1), dim=0)
            else:
                if not self.use_softmax:
                    self._attention = nn.functional.gumbel_softmax(self._alpha, hard=True, dim=0,
                                                                   tau=self._temperature, eps=1e-10)
                else:
                    self._attention = nn.functional.softmax(self._alpha, dim=0)


        else:
            self._alpha.requires_grad = False
            self._attention = self._alpha.data

        if self._alpha.requires_grad and self.retain_grad and self._attention.requires_grad:
            self._attention.retain_grad()

        if DistributedManager.distributed and self._attention.device.type != 'cpu' and self.force_sync_gpu:
            group = DistributedManager.grp
            torch.distributed.broadcast(self._attention, 0, group=group)
        if not self.search_mode:
            if self.multipath_sampling and not self.use_only_alpha:
                self._attention_er = torch.zeros(len(self.exp_ratio), batch_size, dtype=x.dtype, device=x.device)
                self._attention_se = torch.zeros(len(self.se_ratio), batch_size, dtype=x.dtype, device=x.device)
                self._attention_k = torch.zeros(len(self.dw_kernel_size), batch_size, dtype=x.dtype, device=x.device)
            else:
                self._attention_er = torch.zeros(len(self.exp_ratio), dtype=x.dtype, device=x.device)
                self._attention_se = torch.zeros(len(self.se_ratio), dtype=x.dtype, device=x.device)
                self._attention_k = torch.zeros(len(self.dw_kernel_size), dtype=x.dtype, device=x.device)
            # if self._attention_er.device != x.device:
            #     self._attention_er.data = self._attention_er.data.to(x.device)
            #     self._attention_se.data = self._attention_se.data.to(x.device)
            #     self._attention_k.data = self._attention_k.data.to(x.device)
            # if 'HalfTensor' in x.type() and 'HalfTensor' not in self._attention_er.type():
            #     self._attention_er.data = self._attention_er.data.half()
            #     self._attention_se.data = self._attention_se.data.half()
            #     self._attention_k.data = self._attention_k.data.half()

            self.conv_dw_bn.attention = self._attention_k
            self.se_op.attention = self._attention_se
            self.shrinker.attention = self._attention_er

    @property
    def hard_backprop(self):
        return self._hard_backprop

    @hard_backprop.setter
    def hard_backprop(self, val):
        self._hard_backprop = val
        self.update_attention_wrappers()

    def update_attention_wrappers(self, hard_backprob=None):
        for m in self.__dict__.values():
            if isinstance(m, AttentionWrapper):
                m.hard_backprop = self._hard_backprop if hard_backprob is None else hard_backprob

    def __len__(self):
        return len(self.attention)

    def attention_to_param(self):
        index_alpha = 0
        index_max = torch.argmax(self._alpha)
        for e in range(len(self.exp_ratio)):
            for k in range(len(self.dw_kernel_size)):
                for se in range(len(self.se_ratio)):
                    if index_alpha == index_max:
                        return dict(exp_ratio=self.exp_ratio[e], dw_kernel_size=self.dw_kernel_size[k],
                                    se_ratio=self.se_ratio[se])
                    index_alpha += 1

    def alpha_layer_index_to_global_index(self, er, k, se):
        assert er in self.exp_ratio
        assert k in self.dw_kernel_size
        assert se in self.se_ratio
        er_index = self.exp_ratio.index(er)
        k_index = self.dw_kernel_size.index(k)
        se_index = self.se_ratio.index(se)

        index_alpha = 0
        for e in range(len(self.exp_ratio)):
            for k in range(len(self.dw_kernel_size)):
                for se in range(len(self.se_ratio)):
                    if e == er_index and k == k_index and se_index == se:
                        return index_alpha
                    index_alpha += 1
        return None

    def er_k_se_by_attention_index(self, attention_ind):
        assert type(attention_ind) == int
        if self.force_se:
            attention_ind = 2 * attention_ind + 1
        er_index = int(attention_ind / (len(self.dw_kernel_size) + len(self.se_ratio)))
        k_index = int(attention_ind % (len(self.dw_kernel_size) + len(self.se_ratio)) / len(self.se_ratio))
        se_index = int(attention_ind % len(self.se_ratio))
        assert er_index in range(len(self.exp_ratio))
        assert k_index in range(len(self.dw_kernel_size))
        assert se_index in range(len(self.se_ratio))
        return self.exp_ratio[er_index], self.dw_kernel_size[k_index], self.se_ratio[se_index]


class ChannelMasker(nn.Module):
    def __init__(self, factor, chs=None):
        super(ChannelMasker, self).__init__()
        assert 0 <= factor and factor <= 1
        self.factor = factor
        self.chs = chs
        if chs is not None:
            self.mask = self._generate_mask(chs)

    def _generate_mask(self, chs, x=None, divisor=8):
        if not hasattr(self, 'mask') or self.mask is None:
            mid_chs = make_divisible(self.factor * chs, divisor=divisor)
            mask = torch.zeros(1, chs, 1, 1)
            mask[0, :mid_chs, 0, 0] = 1
        else:
            mask = self.mask

        if x is not None:
            if mask.device != x.device:
                mask = mask.to(x.device)
            if 'HalfTensor' in x.type() and 'HalfTensor' not in mask.type():
                mask = mask.half()
            if hasattr(self, 'mask'):
                self.mask = mask

        return mask

    def forward(self, x):
        chs = x.size(1)
        mask = self._generate_mask(chs, x)

        out = mask * x
        return out


class MaskedSqueezeExcite(SqueezeExcite):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=sigmoid, divisor=1, attention=None, list_in_chs=None,
                 **_):
        super(MaskedSqueezeExcite, self).__init__(in_chs, se_ratio, reduced_base_chs, act_layer, gate_fn, divisor)
        self._attention_er = attention
        self.list_in_chs = list_in_chs
        if self.list_in_chs is None:
            self.list_in_chs = [in_chs]
        mid_chs = make_divisible(in_chs * se_ratio)
        # Point-wise expansion
        factors = [float(in_ch * se_ratio) / mid_chs for in_ch in self.list_in_chs if in_ch < max(self.list_in_chs)]
        object.__setattr__(self, 'shrinker', WidthShrinker(self._attention_er, factors, mid_chs))

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.shrinker(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x


class AttentionWrapper(nn.Module):
    def __init__(self, attention, list_modules, hard_backprop=False, chain=False):
        super(AttentionWrapper, self).__init__()
        assert len(list_modules) == len(attention) or (len(list_modules) >= len(attention) and chain)
        self.attention = attention
        self.fixed_modules = None
        num_fixed_modules = 0
        self.list_modules_all = list_modules
        if len(list_modules) > len(attention):
            num_fixed_modules = len(list_modules) - len(attention)
            self.fixed_modules = list_modules[:num_fixed_modules]
        self.list_modules = list_modules[num_fixed_modules:]
        self.hard_backprop = hard_backprop
        self.chain = chain

    def forward(self, x):
        out = 0
        if self.fixed_modules is not None:
            for m in self.fixed_modules:
                x = m(x)
        if (not self.chain) and self.hard_backprop and not len(self.attention[0].shape) > 0:
            return self.list_modules[np.argmax(self.attention.detach().cpu().numpy())](x)
        for a, m in zip(self.attention, self.list_modules):
            o = 0
            if self.chain or (not self.hard_backprop or torch.any(a > 0)):
                o = m(x)
            if not self.hard_backprop or torch.any(a > 0):
                if len(a.shape) > 0:
                    a = a.view(-1, 1, 1, 1)
                out += a * o
            x = o if self.chain else x

        return out

    def __len__(self):
        return len(self.list_modules_all)

    def __getitem__(self, item):
        return self.list_modules_all[item]


class SinkPoint(AttentionWrapper):
    def __init__(self, attention, list_modules, hard_backprop=False):
        super(SinkPoint, self).__init__(attention, list_modules, hard_backprop=hard_backprop, chain=True)
        self._beta_attention_grad = None

    @property
    def beta_attention(self):
        return self.attention

    @beta_attention.setter
    def beta_attention(self, attention):
        self.attention = attention

    @property
    def beta_attention_grad(self):
        if self._beta_attention_grad is not None:
            return self._beta_attention_grad
        return self.attention.grad

    @beta_attention_grad.setter
    def beta_attention_grad(self, grad):
        self._beta_attention_grad = grad


class WidthShrinker(AttentionWrapper):
    def __init__(self, attention, factors=None, chs=None, hard_backprop=False):
        self.factors = factors
        if self.factors is None:
            assert chs is not None
            self.factors = [float(ch) / chs for ch in range(1, chs + 1)]

        module_list = [ChannelMasker(factor, chs) for factor in self.factors] + [nn.Identity()]
        super(WidthShrinker, self).__init__(attention, module_list, hard_backprop)


def _gen_mobilenasnet(channel_multiplier=1.0, exp_ratio=EXP_RATIO_EXTENDED, dw_k_size=DW_K_SIZE, se_ratio=SE_RATIO,
                      heaviest_network=False, use_kernel_3=False, reduced_exp_ratio=False, use_dedicated_pwl_se=False,
                      force_sync_gpu=False, exp_r=6, depth=4, no_privatized_bn=False, multipath_sampling=False,
                      use_softmax=False, search_mode=True, no_swish=False, mobilenet_string='', use_swish=False,
                      **kwargs):
    """Creates a MobileNet-V3 model.

    Ref impl: ?
    Paper: https://arxiv.org/abs/1905.02244

    Args:
      channel_multiplier: multiplier to number of channels per layer.
    """
    num_features = 1280
    if reduced_exp_ratio:
        exp_ratio = EXP_RATIO
        print(f"Using reduced exp ratio: {exp_ratio}")
    if not use_swish:
        act_layer = resolve_act_layer(kwargs, 'hard_swish')
    else:
        print("Using swish")
        act_layer = resolve_act_layer(kwargs, 'swish')
    kernel = 3 if use_kernel_3 else 5
    if no_swish:
        print("Using only ReLU and not h-swish")
    arch_def = [
        # stage 0, 112x112 in
        ['ds_r1_k3_s1_e1_c16_nre'],  # relu
        # stage 1, 112x112 in
        [f'ir_r{depth}_k{kernel}_s2_e{exp_r}_c24_se0.25_nre'],  # relu
        # stage 2, 56x56 in
        [f'ir_r{depth}_k{kernel}_s2_e{exp_r}_c40_se0.25_nre'],  # relu
        # stage 3, 28x28 in
        [f'ir_r{depth}_k{kernel}_s2_e{exp_r}_c80_se0.25' + ('_nre' if no_swish else '')],  # hard-swish
        # stage 4, 14x14in
        [f'ir_r{depth}_k{kernel}_s1_e{exp_r}_c112_se0.25' + ('_nre' if no_swish else '')],  # hard-swish
        # stage 5, 14x14in
        [f'ir_r{depth}_k{kernel}_s2_e{exp_r}_c192_se0.25' + ('_nre' if no_swish else '')],  # hard-swish
        # stage 6, 7x7 in
        ['cn_r1_k1_s1_c960' + ('_nre' if no_swish else '')],  # hard-swish
    ]
    channel_multiplier_stem = channel_multiplier
    if mobilenet_string != '':
        print("Building model from string")
        heaviest_network = True
        arch_def = mobilenet_string.split('], [')
        for i, e in enumerate(arch_def):
            arch_def[i] = e.replace('[', '').replace(']', '').replace('\'', '').split(', ')
            if no_swish:
                arch_def[i] = e.replace('[', '').replace(']', '').replace('\'', '').replace('_nre', '').split(', ')
                for j, k in enumerate(arch_def[i]):
                    arch_def[i][j] = k + '_nre'
        channel_multiplier = 1

    model_kwargs = dict(
        block_args=decode_arch_def(arch_def),
        num_features=num_features,
        stem_size=32,
        channel_multiplier=channel_multiplier,
        channel_multiplier_stem=channel_multiplier_stem,
        norm_kwargs=resolve_bn_args(kwargs),
        act_layer=act_layer,
        se_kwargs=dict(act_layer=nn.ReLU, gate_fn=hard_sigmoid, reduce_mid=True, divisor=8),
        **kwargs,
    )
    if heaviest_network:
        model = MobileNetV3(**model_kwargs)
        return model

    model = MobileNasNet(deepcopy(model_kwargs), dw_k_size, exp_ratio, se_ratio,
                         use_dedicated_pwl_se=use_dedicated_pwl_se, force_sync_gpu=force_sync_gpu,
                         se_kwargs=model_kwargs['se_kwargs'], use_privatized_bn=not no_privatized_bn,
                         multipath_sampling=multipath_sampling, use_softmax=use_softmax, search_mode=search_mode)
    return model


class MobileNasNet(MobileNetV3):
    def __init__(self, model_kwargs, dw_k_size, exp_ratio, se_ratio, use_dedicated_pwl_se, force_sync_gpu,
                 use_privatized_bn, se_kwargs, multipath_sampling, use_softmax, search_mode):
        self.dw_k_size = dw_k_size
        self.exp_ratio = exp_ratio
        self.se_ratio = se_ratio
        self.use_dedicated_pwl_se = use_dedicated_pwl_se
        self.multipath_sampling = multipath_sampling
        self.use_softmax = use_softmax
        super(MobileNasNet, self).__init__(**model_kwargs)
        superblock = partial(InvertedResidualElastic, dw_kernel_size=dw_k_size, exp_ratio=exp_ratio, se_ratio=se_ratio,
                             se_kwargs=se_kwargs, use_dedicated_pwl_se=use_dedicated_pwl_se,
                             force_sync_gpu=force_sync_gpu, use_privatized_bn=use_privatized_bn,
                             multipath_sampling=multipath_sampling, use_softmax=use_softmax, search_mode=search_mode)
        self.hard_backprop = False
        self.force_sync_gpu = force_sync_gpu
        for n, m in self.named_modules():
            if isinstance(m, InvertedResidual):
                c_in = m.conv_pw.in_channels
                c_out = m.conv_pwl.out_channels
                stride = m.conv_dw.stride[0]
                use_relu = isinstance(m.act1, nn.ReLU)
                if use_relu:
                    act = nn.ReLU
                else:
                    act = HardSwishMe

                new_block = superblock(in_chs=c_in, out_chs=c_out, act_layer=act, stride=stride)

                set_layer(self, n, new_block)

        beta_lst = []
        self.beta_attention = []
        self.use_only_beta = False
        self.retain_grad = True
        self._temperature = 1
        self.dict_time = None

        sink_points = []
        for b in self.blocks:
            if len(b) <= 1:
                sink_points.append(b)
                continue
            beta = torch.ones(len(b) - 1) / (len(b) - 1)
            beta_lst.append(torch.nn.parameter.Parameter(beta, requires_grad=False))
            self.beta_attention.append(beta.clone().detach())
            sink_points.append(SinkPoint(self.beta_attention[-1], b))

        self.beta = nn.ParameterList(beta_lst)
        object.__setattr__(self, 'sink_points', nn.Sequential(*sink_points))

    def extract_alpha(self):
        dict_alpha = {}
        for n, m in self.named_parameters():
            if '_alpha' in n:
                layer = '.'.join(n.split('.')[:-1])
                if layer in dict_alpha:
                    dict_alpha[layer].append((n.split('.')[-1], m))
                else:
                    dict_alpha[layer] = [(n.split('.')[-1], m)]

        return dict_alpha

    def set_require_grad(self, val=True):
        self.zero_grad()
        for n, m in self.named_parameters():
            if '_alpha' in n or 'beta' in n:
                m.requires_grad = not val
            else:
                m.requires_grad = val

    def forward_features(self, x):
        self.update_beta(int(x.shape[0]))
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.sink_points(x)
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        return x

    def update_beta(self, batch_size):
        sink_points = [stage for stage in self.sink_points if isinstance(stage, SinkPoint)]
        assert len(sink_points) == len(self.beta_attention) and len(sink_points) == len(self.beta)

        for i, (sink_point, attention, beta) in enumerate(zip(sink_points, self.beta_attention, self.beta)):
            if not self.use_only_beta:
                if attention.device != beta.device:
                    attention.data = attention.data.to(beta.device)
                if 'HalfTensor' in beta.type() and 'HalfTensor' not in attention.type():
                    attention.data = attention.half()
                if self.multipath_sampling:
                    if not self.use_softmax:
                        self.beta_attention[i] = nn.functional.gumbel_softmax(
                            beta.repeat(batch_size).reshape(batch_size, len(beta)).transpose(0, 1),
                            hard=True, dim=0, tau=self._temperature, eps=1e-10)
                    else:
                        self.beta_attention[i] = nn.functional.softmax(
                            beta.repeat(batch_size).reshape(batch_size, len(beta)).transpose(0, 1), dim=0)

                    # self.beta_attention[i] = torch.stack(
                    #     [nn.functional.gumbel_softmax(beta, hard=True, dim=0, tau=self._temperature, eps=1e-10)
                    #     for _ in range(batch_size)]).transpose(0, 1)
                else:
                    if not self.use_softmax:
                        self.beta_attention[i] = nn.functional.gumbel_softmax(beta, hard=True, dim=0,
                                                                              tau=self._temperature, eps=1e-10)
                    else:
                        self.beta_attention[i] = nn.functional.softmax(beta, dim=0)
                sink_point.attention = self.beta_attention[i]
            else:
                beta.requires_grad = False
                self.beta_attention[i] = beta.data
                sink_point.attention.data = self.beta_attention[i]

            if DistributedManager.distributed and self.beta_attention[i].device.type != 'cpu' and self.force_sync_gpu:
                group = DistributedManager.grp
                torch.distributed.broadcast(self.beta_attention[i], 0, group=group)

            if beta.requires_grad and self.retain_grad and self.beta_attention[i].requires_grad:
                self.beta_attention[i].retain_grad()

    def forward(self, x):
        x = self.forward_features(x)
        x = x.flatten(1)
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return self.classifier(x)

    def set_use_only_alpha(self, val=True):
        for n, m in self.named_modules():
            if hasattr(m, 'use_only_alpha'):
                m.use_only_alpha = val

    def set_retain_grad(self, val=True):
        for n, m in self.named_modules():
            if hasattr(m, '_attention'):
                m.retain_grad = val
        for b in self.beta_attention:
            b.retain_grad = val

    def load_from_super_network(self, super_model):
        super_state_dict = super_model.state_dict()
        state_dict = self.state_dict()
        new_state_dict = {}

        key_orig = list(state_dict.keys())
        for k, v in state_dict.items():
            if '_alpha' in k or 'beta' in k:
                continue
            k_splitted = k.split('.')
            old_key = '.'.join(k_splitted[:4] + k_splitted[5:])
            assert k in super_state_dict or old_key in super_state_dict
            if k in super_state_dict:
                new_state_dict[k] = super_state_dict[k]
                key_orig.remove(k)
            else:
                assert 'conv_pw' not in k or 'conv_pwl' in k
                assert 'bn1' not in k
                assert 'bn3' not in k or layer.shape == superlayer.shape
                assert 'se' not in k or layer.shape == superlayer.shape

                layer = extract_layer(self, k)
                superlayer = extract_layer(super_model, old_key)
                if layer.shape == superlayer.shape:
                    key_orig.remove(k)
                    new_state_dict[k] = super_state_dict[old_key]
        if DistributedManager.is_master():
            logging.info(f"Missing keys are : {key_orig}")
        self.load_state_dict(new_state_dict, strict=False)

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, val):
        if val is None:
            return

        self._temperature = val
        for n, m in self.named_modules():
            if hasattr(m, '_alpha') or hasattr(m, '_beta'):
                m.temperature = val

    def set_uniform_alpha(self, prefer_higher_width_fact=1.0, prefer_higher_k_fact=1.0):
        for m in self.modules():
            if hasattr(m, '_alpha'):
                len_alpha = len(m._alpha)
                new_alpha = torch.tensor([1.0] * len_alpha) / len_alpha
                new_alpha = new_alpha.to(m._alpha.device)
                if prefer_higher_width_fact > 1.0 or prefer_higher_k_fact > 1.0:
                    for e, k in itertools.product(self.exp_ratio, self.dw_k_size):
                        factor = 1.0
                        if e == self.exp_ratio[-1]:
                            factor *= prefer_higher_width_fact
                        if k == self.dw_k_size[-1]:
                            factor *= prefer_higher_k_fact
                        if factor > 1:
                            index1 = m.alpha_layer_index_to_global_index(er=e, k=k,
                                                                         se=self.se_ratio[-1])
                            new_alpha.data[index1] = factor / len_alpha
                if 'HalfTensor' in m._alpha.type():
                    new_alpha = new_alpha.half()
                if m.force_se:
                    new_alpha[0::2] = -float("inf")
                m._alpha.data = new_alpha
                m.use_only_alpha = False

    def set_uniform_beta(self, stage=0, prefer_higher_beta_fact=1.0):
        for m in self.modules():
            if hasattr(m, 'beta'):
                for i, beta in enumerate(m.beta):
                    len_beta = len(beta)
                    if len_beta == 1:
                        new_beta = torch.tensor([1.0])
                        new_beta = new_beta.to(beta.device)
                    else:
                        new_beta = torch.tensor([1.0] * len_beta) / (len_beta)
                        if stage == 0:
                            new_beta[0] = -1000.0
                        new_beta = new_beta.to(beta.device)
                        if prefer_higher_beta_fact > 1.0:
                            new_beta[-1] = prefer_higher_beta_fact / (len_beta)
                    if 'HalfTensor' in beta.type():
                        new_beta = new_beta.half()
                    m.beta[i].data = new_beta
                    m.beta[i].requires_grad_(False)
                m.use_only_beta = False

    def set_argmax_alpha_beta(self, use_only=True, only_beta=False):
        for m in self.modules():
            if hasattr(m, '_alpha') and not only_beta:
                # m.use_only_alpha = use_only
                alpha = m._alpha
                argmax = torch.argmax(alpha)
                # alpha.data *= 0
                alpha.data -= float('inf')
                alpha.data[argmax] = 1
                alpha.requires_grad_(not use_only)
                # Generate the corresponding attention
                _ = m.attention

            if hasattr(m, 'beta'):
                # m.use_only_beta = use_only
                for i, beta in enumerate(m.beta):
                    argmax = torch.argmax(beta)
                    # beta.data *= 0
                    beta.data -= float('inf')
                    beta.data[argmax] = 1
                    beta.requires_grad_(not use_only)

                m.update_beta(batch_size=1)

    def set_last_alpha(self, use_only=True):
        self.set_all_alpha(er=self.exp_ratio[-1], k=self.dw_k_size[-1], se=self.se_ratio[-1], use_only=use_only)

    def set_all_alpha(self, er, k, se, use_only=True):
        for m in self.modules():
            if hasattr(m, '_alpha'):
                index = m.alpha_layer_index_to_global_index(er, k, se)
                len_alpha = len(m._alpha)
                new_alpha = torch.tensor([0.0] * len_alpha)
                new_alpha[index] = 1.0
                new_alpha = 1000 * (new_alpha - 1) if not use_only else new_alpha
                new_alpha = new_alpha.to(m._alpha.device)
                if 'HalfTensor' in m._alpha.type():
                    new_alpha = new_alpha.half()
                m._alpha.data = new_alpha
                m.use_only_alpha = use_only
                m._alpha.requires_grad_(not use_only)

    def set_all_beta(self, num_layer=-1, use_only=True):
        if num_layer != -1:
            num_layer -= 1
        for m in self.modules():
            if hasattr(m, 'beta'):
                for i, beta in enumerate(m.beta):
                    len_beta = len(beta)
                    new_beta = torch.tensor([0.0] * len_beta)
                    index = num_layer - 1
                    if num_layer == -1:
                        index = -1
                    new_beta[index] = 1.0
                    new_beta = 1000 * (new_beta - 1) if not use_only else new_beta
                    new_beta = new_beta.to(beta.device)
                    if 'HalfTensor' in beta.type():
                        new_beta = new_beta.half()
                    m.beta[i].data = new_beta
                    m.beta[i].requires_grad_(not use_only)
                m.use_only_beta = use_only

    def set_last_beta(self, use_only=True):
        self.set_all_beta(use_only=use_only)

    def set_width_training(self, first_stage=False, prefer_higher_width_fact=1.0):
        self.set_last_alpha()
        self.set_last_beta()
        self.requires_grad_(True)
        self.train(True)
        exp_ratio = self.exp_ratio
        if first_stage:
            exp_ratio = self.exp_ratio[-2:]
        for m in self.modules():
            if hasattr(m, '_alpha'):
                alpha_len = len(m._alpha)
                initial_alpha = -1000 * torch.ones(alpha_len).to(m._alpha.device)
                if 'HalfTensor' in m._alpha.type():
                    initial_alpha = initial_alpha.half()
                for i, e in enumerate(exp_ratio):
                    index_k = m.alpha_layer_index_to_global_index(e, self.dw_k_size[-1], self.se_ratio[-1])
                    c = 1.0
                    if prefer_higher_width_fact > 1.0 and i == len(exp_ratio) - 1:
                        c = prefer_higher_width_fact
                    initial_alpha[index_k] = c / len(exp_ratio)
                m._alpha.data = initial_alpha
                m.use_only_alpha = False
                m._alpha.requires_grad_(False)
        for p in self.beta:
            p.requires_grad_(False)

        for n, m in self.named_modules():
            if 'conv_dw' in n and isinstance(m, nn.Conv2d) and m.kernel_size[0] != self.dw_k_size[-1] and \
                n.split('.')[
                    -1] != 'conv_dw':
                m.requires_grad_(False)
            if 'bn2' in n and isinstance(m, nn.BatchNorm2d) and n.split('.')[-1] != 'bn2' and int(
                n.split('.')[-1]) != len(self.dw_k_size) - 1:
                m.requires_grad_(False)
                m.train(False)

            if 'bn3' in n and isinstance(m, nn.BatchNorm2d) and n.split('.')[-1] != 'bn3' and int(
                n.split('.')[-1]) % len(self.se_ratio) != len(self.se_ratio) - 1:
                m.requires_grad_(False)
                m.train(False)

            if self.use_dedicated_pwl_se:
                if 'conv_pwl' in n and isinstance(m, nn.Conv2d) and n.split('.')[-1] != 'conv_pwl' and int(
                    n.split('.')[-1]) != len(self.se_ratio) - 1:
                    m.requires_grad_(False)

    def set_only_kernel_training(self, also_last=True, afterwidth=True):
        self.set_kernel_training(also_last=also_last, afterwidth=afterwidth)
        self.train(False)
        self.requires_grad_(False)
        for n, m in self.named_modules():
            if 'conv_dw' in n and isinstance(m, nn.Conv2d) and m.kernel_size[0] != self.dw_k_size[-1] and n.split('.')[
                -1] != 'conv_dw':
                m.requires_grad_(True)
            if 'bn2' in n and isinstance(m, nn.BatchNorm2d) and n.split('.')[-1] != 'bn2' and int(
                n.split('.')[-1]) != len(self.dw_k_size) - 1:
                m.requires_grad_(True)
                m.train(True)

    def set_kernel_training(self, also_last=True, afterwidth=True, prefer_higher_width_fact=1.0,
                            prefer_higher_k_fact=1.0):
        self.set_last_alpha()
        self.set_last_beta()
        self.requires_grad_(True)
        self.train(True)
        dw_k_size = self.dw_k_size
        exp_ratio = self.exp_ratio
        if not also_last:
            dw_k_size = self.dw_k_size[:-1]
        if not afterwidth:
            exp_ratio = [self.exp_ratio[-1]]
        for n, m in self.named_modules():
            if hasattr(m, '_alpha'):
                alpha_len = len(m._alpha)
                initial_alpha = -1000 * torch.ones(alpha_len).to(m._alpha.device)
                if 'HalfTensor' in m._alpha.type():
                    initial_alpha = initial_alpha.half()
                for j, k in enumerate(dw_k_size):
                    for i, e in enumerate(exp_ratio):
                        c = 1.0
                        if prefer_higher_width_fact > 1 and i == len(exp_ratio) - 1:
                            c *= prefer_higher_width_fact
                        if prefer_higher_k_fact > 1 and j == len(dw_k_size) - 1:
                            c *= prefer_higher_k_fact
                        index_k = m.alpha_layer_index_to_global_index(e, k, self.se_ratio[-1])
                        initial_alpha[index_k] = c / (len(dw_k_size) * len(exp_ratio))
                m._alpha.data = initial_alpha
                m.use_only_alpha = False
                m._alpha.requires_grad_(False)

            if 'bn3' in n and isinstance(m, nn.BatchNorm2d) and n.split('.')[-1] != 'bn3' and int(
                n.split('.')[-1]) % len(self.se_ratio) != len(self.se_ratio) - 1:
                m.requires_grad_(False)
                m.train(False)

            if self.use_dedicated_pwl_se:
                if 'conv_pwl' in n and isinstance(m, nn.Conv2d) and n.split('.')[-1] != 'conv_pwl' and int(
                    n.split('.')[-1]) != len(self.se_ratio) - 1:
                    m.requires_grad_(False)

        for p in self.beta:
            p.requires_grad_(False)

    def set_hard_backprop(self, val=True):
        for n, m in self.named_modules():
            if hasattr(m, 'hard_backprop'):
                m.hard_backprop = val

        for n, m in self.sink_points.named_modules():
            if hasattr(m, 'hard_backprop'):
                m.hard_backprop = val

    def set_force_se(self, val=False):
        for m in self.modules():
            if isinstance(m, InvertedResidualElastic):
                m.force_se = val

    def extract_expected_latency(self, batch_size=16, iterations=20, file_name='lut.pkl',
                                 target='cpu'):
        if self.dict_time is None:
            self.dict_time = generate_latency_dict(self, file_name=file_name, batch_size=batch_size,
                                                   iterations=iterations, target=target)
        latency = 0
        for n, m in self.named_modules():
            if isinstance(m, InvertedResidualElastic):
                curr_alpha_index = torch.argmax(m.alpha)
                stage_index = int(n.split('blocks.')[1][0])
                index_layer = int(n.split('blocks.')[1][2])
                curr_beta = self.beta[stage_index - 1]
                last_layer_to_take = torch.argmax(curr_beta).item() + 1
                if index_layer <= last_layer_to_take:
                    latency += self.dict_time[n][curr_alpha_index]
        latency += self.dict_time['general']
        return latency


class IKD_mobilenasnet_model(nn.Module):
    def __init__(self, module1, module2, alpha_const, ikd_dividor=10, real_KD=False):
        super(IKD_mobilenasnet_model, self).__init__()
        self.alpha_const = alpha_const
        self.real_KD = real_KD
        self.ikd_dividor = ikd_dividor
        module1.eval()
        module1.requires_grad_(False)
        object.__setattr__(self, 'module1', module1)
        object.__setattr__(self, 'module2', module2)

        for n, m in self.module2.named_modules():
            if n != '' and '.' not in n:
                self.add_module(n, m)

    def forward(self, x):
        self.module2.update_beta(int(x.shape[0]))
        self.module1.eval()
        if self.training:
            with torch.no_grad():
                x1 = self.module1.conv_stem(x)
                x1 = self.module1.bn1(x1)
                x1 = self.module1.act1(x1)

        x2 = self.module2.conv_stem(x)
        x2 = self.module2.bn1(x2)
        x2 = self.module2.act1(x2)
        loss = 0
        if self.training and self.ikd_dividor > 0:
            loss = torch.mean((x2 - x1) ** 2) / self.ikd_dividor

        assert len(self.module1.blocks) == len(self.module2.sink_points)

        for m1, m2 in zip(self.module1.blocks, self.module2.sink_points):
            assert len(m1) == len(m2)
            x2 = m2(x2)
            if self.training:
                with torch.no_grad():
                    x1 = m1(x1)
                if self.ikd_dividor > 0:
                    loss += torch.mean((x2 - x1) ** 2) / self.ikd_dividor
        if self.training:
            with torch.no_grad():
                x1 = self.module1.global_pool(x1)
                x1 = self.module1.conv_head(x1)
                x1 = self.module1.act2(x1)
                x1 = x1.flatten(1)
                out1 = self.module1.classifier(x1)

        x2 = self.module2.global_pool(x2)
        x2 = self.module2.conv_head(x2)
        x2 = self.module2.act2(x2)
        x2 = x2.flatten(1)
        out2 = self.module2.classifier(x2)
        if self.training:
            if not self.real_KD:
                loss += torch.mean((out2 - out1) ** 2)
            else:
                target = F.softmax(out1, dim=-1)
                input = F.log_softmax(out2, dim=-1)
                loss += F.kl_div(input, target, reduction='batchmean')
            loss *= self.alpha_const
            return out2, loss
        return out2

    def train(self, mode=True):
        super(IKD_mobilenasnet_model, self).train(mode=mode)
        self.module1.eval()
        self.module2.train(mode)
        return self

    def half(self):
        super(IKD_mobilenasnet_model, self).half()
        self.module1.half()
        self.module2.half()
        return self

    def float(self):
        super(IKD_mobilenasnet_model, self).float()
        self.module1.float()
        self.module2.float()
        return self

    def cpu(self):
        super(IKD_mobilenasnet_model, self).cpu()
        self.module1.cpu()
        self.module2.cpu()
        return self

    def cuda(self, device=None):
        super(IKD_mobilenasnet_model, self).cuda(device=device)
        self.module1.cuda(device=device)
        self.module2.cuda(device=device)
        return self

    def to(self, *args, **kwargs):
        super(IKD_mobilenasnet_model, self).to(*args, **kwargs)
        self.module1.to(*args, **kwargs)
        self.module2.to(*args, **kwargs)
        return self


def compute_taylor(model, data_loader, loss=nn.CrossEntropyLoss().cuda()):
    model.zero_grad()
    model.eval()
    layers_list = extract_conv_layers(model)
    if data_loader is not None:
        len_data_loader = len(data_loader)
        for batch_idx, (input, target) in enumerate(data_loader):
            input, target = input.cuda(), target.cuda()
            output = model(input)
            l = loss(output, target) / len_data_loader
            l.backward()
            if batch_idx % 10 == 0 and DistributedManager.is_master():
                logging.info(f"Computing gradient for weight importance estimation: Done {batch_idx}/{len_data_loader}")
            del l, input, target, output
            gc.collect()
            torch.cuda.empty_cache()
    taylor_per_layer = {}
    model.cpu()
    for layer in layers_list:
        layer_ = extract_layer(model, layer)
        if data_loader is not None:
            taylor_per_layer[layer] = torch.sum(torch.abs(layer_.weight * layer_.weight.grad), dim=[1, 2, 3]).cuda()
        else:
            taylor_per_layer[layer] = torch.sum(torch.abs(layer_.weight), dim=[1, 2, 3]).cuda()

        if DistributedManager.distributed:
            grp = DistributedManager.grp
            ws = torch.distributed.get_world_size()
            torch.distributed.all_reduce(taylor_per_layer[layer], op=torch.distributed.ReduceOp.SUM, group=grp)
            taylor_per_layer[layer] /= ws
        taylor_per_layer[layer] = taylor_per_layer[layer].float().detach().cpu().numpy()
    DistributedManager.set_barrier()
    return taylor_per_layer


def update_training_mode(model, mode='width', prefer_higher_width_fact=1.0, prefer_higher_k_fact=1.0,
                         prefer_higher_beta_fact=1.0):
    if mode == 'static':
        if DistributedManager.is_master():
            logging.info("Training mode: Static")
        model.set_last_alpha()
        model.set_last_beta()

    if mode == 'width':
        if DistributedManager.is_master():
            logging.info("Training mode: Elastic Width")
        model.set_width_training(prefer_higher_width_fact=prefer_higher_width_fact)

    if mode == 'kernel1':
        if DistributedManager.is_master():
            logging.info("Training mode: Low Kernel Only")
        model.set_only_kernel_training(also_last=False)

    if mode == 'kernel2':
        if DistributedManager.is_master():
            logging.info("Training mode: Fixed Kernel Only")
        model.set_only_kernel_training()

    if mode == 'kernel3':
        if DistributedManager.is_master():
            logging.info("Training mode: Elastic Kernel Size and Width ")
        model.set_kernel_training(prefer_higher_width_fact=prefer_higher_width_fact,
                                  prefer_higher_k_fact=prefer_higher_k_fact)

    if mode == 'depth0':
        if DistributedManager.is_master():
            logging.info("Training mode: Elastic Kernel Size, width and two last depth")
        model.set_kernel_training(prefer_higher_width_fact=prefer_higher_width_fact,
                                  prefer_higher_k_fact=prefer_higher_k_fact)
        model.set_uniform_beta(prefer_higher_beta_fact=prefer_higher_beta_fact)

    if mode == 'depth1':
        if DistributedManager.is_master():
            logging.info("Training mode: Elastic Kernel Size, width and depth")
        model.set_kernel_training(prefer_higher_width_fact=prefer_higher_width_fact,
                                  prefer_higher_k_fact=prefer_higher_k_fact)
        model.set_uniform_beta(stage=1, prefer_higher_beta_fact=prefer_higher_beta_fact)

    if mode == 'depth0_no_kernel':
        if DistributedManager.is_master():
            logging.info("Training mode: width and two last depth")
        model.set_width_training(prefer_higher_width_fact=prefer_higher_width_fact)
        model.set_uniform_beta(prefer_higher_beta_fact=prefer_higher_beta_fact)

    if mode == 'depth1_no_kernel':
        if DistributedManager.is_master():
            logging.info("Training mode: width and depth")
        model.set_width_training(prefer_higher_width_fact=prefer_higher_width_fact)
        model.set_uniform_beta(stage=1, prefer_higher_beta_fact=prefer_higher_beta_fact)

    if mode == 'kernel1_depth':
        if DistributedManager.is_master():
            logging.info("Training mode: Low Kernel Only")
        model.set_only_kernel_training(also_last=False)
        model.set_uniform_beta(stage=1, prefer_higher_beta_fact=prefer_higher_beta_fact)

    if mode == 'kernel2_depth':
        if DistributedManager.is_master():
            logging.info("Training mode: Fixed Kernel Only")
        model.set_only_kernel_training()
        model.set_uniform_beta(stage=1, prefer_higher_beta_fact=prefer_higher_beta_fact)

    if mode == 'se':
        if DistributedManager.is_master():
            logging.info("Training mode: Elastic Kernel Size, width and SE")
        model.set_require_grad()
        model.set_uniform_alpha(prefer_higher_width_fact=prefer_higher_width_fact,
                                prefer_higher_k_fact=prefer_higher_k_fact)
        model.set_last_beta()
        model.train()

    if mode == 'full0':
        if DistributedManager.is_master():
            logging.info("Training mode: Elastic Kernel Size, width, two last depth and SE")
        model.set_require_grad()
        model.set_uniform_alpha(prefer_higher_width_fact=prefer_higher_width_fact,
                                prefer_higher_k_fact=prefer_higher_k_fact)
        model.set_uniform_beta(prefer_higher_beta_fact=prefer_higher_beta_fact)

    if mode == 'full1':
        if DistributedManager.is_master():
            logging.info("Training mode: Elastic Kernel Size, width, depth and SE")
        model.set_require_grad()
        model.set_uniform_alpha(prefer_higher_width_fact=prefer_higher_width_fact,
                                prefer_higher_k_fact=prefer_higher_k_fact)
        model.set_uniform_beta(stage=1, prefer_higher_beta_fact=prefer_higher_beta_fact)


def reorganize_channels(model, dataset, args):
    taylor_per_layer_file = os.path.join(args.output, 'taylor_per_layer.pkl')
    model.eval()
    model.cuda()
    x = torch.rand(10, 3, 224, 224).cuda()
    out1 = model(x)

    if not os.path.exists(taylor_per_layer_file):
        data_config = resolve_data_config(vars(args), model=model, verbose=False)
        data_loader = create_loader(
            dataset,
            input_size=data_config['input_size'],
            batch_size=args.batch_size // 2,
            is_training=False,
            use_prefetcher=args.prefetcher,
            interpolation=data_config['interpolation'],
            mean=data_config['mean'],
            std=data_config['std'],
            num_workers=args.workers,
            distributed=args.distributed,
            collate_fn=None,
            crop_pct=data_config['crop_pct'],
            pin_memory=args.pin_mem,
            squish=args.squish,
            infinite_loader=False,
        )
        taylor_per_layer = compute_taylor(model, data_loader)
        if DistributedManager.is_master():
            with open(taylor_per_layer_file, 'wb') as f:
                pickle.dump(taylor_per_layer, f)

            logging.info(f"Taylor per layer list saved in {taylor_per_layer_file}")
    else:
        with open(taylor_per_layer_file, 'rb') as f:
            taylor_per_layer = pickle.load(f)

        if DistributedManager.is_master():
            logging.info(f"Taylor per layer loaded from {taylor_per_layer_file}")

    model.eval()
    model.zero_grad()
    model.cpu()
    for k, v in taylor_per_layer.items():
        if k.split('.')[-1] == 'conv_pw':
            layer = extract_layer(model, '.'.join(k.split('.')[:-1]))
            if not isinstance(layer, InvertedResidual):
                continue
            prefix = '.'.join(k.split('.')[:-1])
            v += taylor_per_layer[prefix + '.' + 'conv_dw']
            v += taylor_per_layer[prefix + '.' + 'se.conv_expand']
            order = torch.LongTensor(list(np.argsort(v)[::-1]))
            suffix = '.'.join(k.split('.')[:-1]) + '.'
            conv_pw = extract_layer(model, suffix + 'conv_pw')
            conv_dw = extract_layer(model, suffix + 'conv_dw')
            bn1 = extract_layer(model, suffix + 'bn1')
            bn2 = extract_layer(model, suffix + 'bn2')
            se_conv_reduce = extract_layer(model, suffix + 'se.conv_reduce')
            se_conv_expand = extract_layer(model, suffix + 'se.conv_expand')
            conv_pwl = extract_layer(model, suffix + 'conv_pwl')

            conv_pw.weight.data = conv_pw.weight.data[order, :, :, :].clone()
            bn1.weight.data = bn1.weight.data[order].clone()
            bn1.bias.data = bn1.bias.data[order].clone()
            bn1.running_mean.data = bn1.running_mean.data[order].clone()
            bn1.running_var.data = bn1.running_var.data[order].clone()

            conv_dw.weight.data = conv_dw.weight.data[order, :, :, :].clone()
            bn2.weight.data = bn2.weight.data[order].clone()
            bn2.bias.data = bn2.bias.data[order].clone()
            bn2.running_mean.data = bn2.running_mean.data[order].clone()
            bn2.running_var.data = bn2.running_var.data[order].clone()

            conv_pwl.weight.data = conv_pwl.weight.data[:, order, :, :].clone()
            se_conv_reduce.weight.data = se_conv_reduce.weight.data[:, order, :, :].clone()
            se_conv_expand.weight.data = se_conv_expand.weight.data[order, :, :, :].clone()
            se_conv_expand.bias.data = se_conv_expand.bias.data[order].clone()

            order_se = torch.LongTensor(list(np.argsort(taylor_per_layer[prefix + '.' + 'se.conv_reduce'])[::-1]))
            se_conv_reduce.weight.data = se_conv_reduce.weight.data[order_se, :, :, :].clone()
            se_conv_expand.weight.data = se_conv_expand.weight.data[:, order_se, :, :].clone()
            se_conv_reduce.bias.data = se_conv_reduce.bias.data[order_se].clone()

    model.eval()
    model.cuda()
    out2 = model(x)
    eps = 1e-7
    assert (torch.sum((out1 - out2) ** 2)) < eps
    del out1, out2, x
    gc.collect()
    torch.cuda.empty_cache()
    model.cpu()


@register_model
def mobilenasnet(num_classes=1000, in_chans=3, **kwargs):
    """Mobile NasNet.
    """
    kwargs.pop('pretrained', None)
    model = _gen_mobilenasnet(num_classes=num_classes, in_chans=in_chans, **kwargs)

    return model


@register_model
def mobilenasnet_large(num_classes=1000, in_chans=3, **kwargs):
    """Mobile NasNet.
    """
    kwargs.pop('pretrained', None)
    model = _gen_mobilenasnet(channel_multiplier=1.2, num_classes=num_classes, in_chans=in_chans, **kwargs)

    return model


#
def transform_model_to_mobilenet(model, load_weight=True, mobilenet_string='', channel_multiplier=1):
    def _get_code(m):
        assert isinstance(m, InvertedResidualElastic)
        alpha_index = int(torch.argmax(m.alpha).item())
        er, k, se = m.er_k_se_by_attention_index(alpha_index)
        if m.force_se:
            alpha_index = 2 * alpha_index + 1
        assert m.alpha_layer_index_to_global_index(er, k, se) == alpha_index
        se = (se != 0)
        num_ch = m.conv_pwl.out_channels
        stride = m.conv_dw[0].stride[0]
        use_relu = isinstance(m.act1, nn.ReLU)
        template = lambda kernel, stride, exp_r, num_ch, relu, se: \
            f"ir_r1_k{kernel}_s{stride}_e{exp_r}_c{num_ch}{'_nre' if relu else ''}{'_se0.25' if se else ''}"
        return template(k, stride, er, num_ch, use_relu, se)

    if hasattr(model, 'module'):
        model = model.module

    if mobilenet_string == '':
        if model.blocks[0][0].conv_pw.weight.shape[0] == 24:
            channel_multiplier = 1.2
        else:
            channel_multiplier = 1

        if channel_multiplier == 1.2:
            arch_def_init = ['ds_r1_k3_s1_e1_c24_nre']  # relu
            arch_def_end = ['cn_r1_k1_s1_c1152']  # hard-swish
        else:
            arch_def_init = ['ds_r1_k3_s1_e1_c16_nre']  # relu
            arch_def_end = ['cn_r1_k1_s1_c960']  # hard-swish
        has_relu = isinstance(model.blocks[-1][0].act1, nn.ReLU)
        if has_relu:
            arch_def_end = ['cn_r1_k1_s1_c960_nre']
        arch_def = []
        arch_def.append(arch_def_init)
        for i, block in enumerate(model.blocks[1:-1]):
            str_block = []
            assert len(block) == len(model.beta[i]) + 1
            for j, m in enumerate(block):
                assert isinstance(m, InvertedResidualElastic)
                code = _get_code(m)
                # if j == 0 or torch.sum(model.beta[i][(j - 1):]) > 0:
                if j == 0 or torch.argmax(model.beta[i]) >= j - 1:
                    str_block.append(code)
            arch_def.append(str_block)

        arch_def.append(arch_def_end)
    else:
        print("Building model from string")
        arch_def = mobilenet_string.split('], [')
        for i, e in enumerate(arch_def):
            arch_def[i] = e.replace('[', '').replace(']', '').replace('\'', '').split(', ')
    kwargs = {}
    # channel_multiplier = channel_multiplier
    num_features = 1280
    if isinstance(model.act1, HardSwishMe):
        act_layer = resolve_act_layer(kwargs, 'hard_swish')
    else:
        print("Using swish")
        act_layer = resolve_act_layer(kwargs, 'swish')

    model_kwargs = dict(
        block_args=decode_arch_def(arch_def),
        num_features=num_features,
        stem_size=32,
        channel_multiplier=1,
        channel_multiplier_stem=channel_multiplier,
        norm_kwargs=resolve_bn_args(kwargs),
        act_layer=act_layer,
        se_kwargs=dict(act_layer=nn.ReLU, gate_fn=hard_sigmoid, reduce_mid=True, divisor=8),
        **kwargs,
    )
    model2 = MobileNetV3(**model_kwargs)
    if load_weight:
        model2 = load_from_supernetwork(model, model2)
    return model2, arch_def


def load_from_supernetwork(model, model2):
    model_st_dct = model.state_dict()
    model2_st_dct = model2.state_dict()
    for k, v in model2_st_dct.items():
        new_k = k
        if 'conv_dw' in k and k not in model_st_dct:
            size_kernel = v.shape[-1]
            m = extract_layer(model, k.split('conv_dw')[0])
            index_k = m.dw_kernel_size.index(size_kernel)
            k_prefix = k.split('conv_dw')[0] + 'conv_dw'
            k_suffix = k.split('conv_dw')[1]
            new_k = k_prefix + '.' + str(index_k) + k_suffix
            assert new_k in model_st_dct
        elif 'bn2' in k and k not in model_st_dct:
            original_conv = model2_st_dct[k.split('bn2')[0] + 'conv_dw.weight']
            size_kernel = original_conv.shape[-1]
            m = extract_layer(model, k.split('bn2')[0])
            index_k = m.dw_kernel_size.index(size_kernel)
            k_prefix = k.split('bn2')[0] + 'bn2'
            k_suffix = k.split('bn2')[1]
            new_k = k_prefix + '.' + str(index_k) + k_suffix
            assert new_k in model_st_dct
        elif '.se.' in k and k not in model_st_dct:
            k_prefix = k.split('.se.')[0] + '.se.'
            k_suffix = '.' + k.split('.se.')[1]
            new_k = k_prefix + str(1) + k_suffix
            assert new_k in model_st_dct
        elif 'bn3' in k:
            # m = extract_layer(model, k.split('bn3')[0])
            # alpha_index = int(torch.argmax(m.alpha).item())
            # er_, _, se_ = m.er_k_se_by_attention_index(alpha_index)
            m2 = extract_layer(model2, k.split('bn3')[0])
            er = int(m2.conv_pw.weight.shape[0] / m2.conv_pw.weight.shape[1])
            se = 0 if not hasattr(m2, 'se') or m2.se is None else m.se_ratio[-1]
            # if er_ != er or se_ != se:
            #     print(er_, er, se_, se)
            if m.use_privatized_bn:
                index_k = m.exp_ratio.index(er) * len(m.se_ratio) + m.se_ratio.index(se)
            else:
                index_k = 0
            k_prefix = k.split('bn3')[0] + 'bn3'
            k_suffix = k.split('bn3')[1]
            new_k = k_prefix + '.' + str(index_k) + k_suffix
            assert new_k in model_st_dct, new_k
        elif k not in model_st_dct:
            print(k)
            continue
        # print(k, new_k)
        weight = model_st_dct[new_k]
        if weight.shape == v.shape:
            model2_st_dct[k].copy_(weight.clone())
        elif weight.shape[0] != v.shape[0]:
            assert len(weight.shape) == 1 or weight.shape[1:] == v.shape[1:] or '.se.' in k
            if not '.se.' in k or len(v.shape) == 1:
                model2_st_dct[k].copy_(weight[:v.shape[0], ...].clone())
            else:
                model2_st_dct[k].copy_(weight[:v.shape[0], :v.shape[1], ...].clone())
        else:
            assert len(weight.shape) > 1
            assert weight.shape[1] != v.shape[1]
            model2_st_dct[k].copy_(weight[:, :v.shape[1], ...].clone())

    return model2


def extract_module_name(model, module):
    model_ = model
    if hasattr(model_, 'module'):
        model_ = model_.module
    name = [name for name, _module in model_.named_modules() if _module == module][0]
    return name


def compute_resolution(model):
    def _set_resolution_hook(model):
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

    model.eval()
    is_model_on_cuda = 'cuda' in model.beta[0].device.type
    model.cpu()
    resolution_dict, list_hook = _set_resolution_hook(model)
    with torch.no_grad():
        model(torch.rand(10, 3, 224, 224))
    for h in list_hook:
        h.remove()
    if is_model_on_cuda:
        model.cuda()
    return resolution_dict


def measure_time_inverted_bottleneck(desc, batch_size=1, num_iter=400, target='cpu'):
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

    if target == 'onnx':
        if not os.path.exists('./outputs/onnx/'):
            os.mkdir('./outputs/onnx/')
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        model_onnx = f'./outputs/onnx/{desc}.onnx'
        torch.onnx.export(model=module.cpu(), args=input.to('cpu'),
                          f=model_onnx,
                          input_names=['input'], output_names=['output'])  #
        ort_session = ort.InferenceSession(model_onnx)
        _ = ort_session.run(None, {
            'input': np.random.randn(batch_size, in_channels, resolution, resolution).astype(np.float32)})
        total_time = 0
        for i in range(num_iter):
            inp_ = np.random.randn(batch_size, in_channels, resolution, resolution).astype(np.float32)
            start = time.time()
            _ = ort_session.run(None, {'input': inp_})
            end = time.time()
            total_time += end - start

        return total_time / (num_iter * batch_size)

    if target == 'cpu':
        module.cpu()
        torch.set_num_threads(1)

    else:
        assert target == 'gpu', "target_device must be either 'cpu, 'gpu' or 'onnx'"
        module.cuda()
        input = input.cuda()
        start_gpu = torch.cuda.Event(enable_timing=True)
        end_gpu = torch.cuda.Event(enable_timing=True)
        time_all = 0

    with torch.no_grad():
        res = module(input)
    start = time.time()
    with torch.no_grad():
        for i in range(num_iter):
            if target != 'cpu':
                start_gpu.record()
            _ = module(input)
            if target != 'cpu':
                end_gpu.record()
                torch.cuda.synchronize()
                time_all += start_gpu.elapsed_time(end_gpu) / 1000.0

    end = time.time()
    del module, input, res
    gc.collect()
    if target != 'cpu':
        return time_all / num_iter
    return (end - start) / (num_iter * batch_size)


def extract_resolution_stride_dict(model):
    desc_blocks = {}
    resolution = compute_resolution(model)
    for n, m in model.named_modules():
        if isinstance(m, InvertedResidualElastic):
            stride = m.conv_dw[0].stride[0]
            in_ch = m.conv_pw.in_channels
            out_ch = m.conv_pwl.out_channels if isinstance(m.conv_pwl, nn.Conv2d) else \
                m.conv_pwl[0].out_channels
            res = resolution[extract_module_name(model, m.conv_pw)]
            use_relu = int(isinstance(m.act1, nn.ReLU))
            list_possible_params = []
            for index in range(len(m.alpha)):
                # print(index)
                # print(m.er_k_se_by_attention_index(index))
                list_possible_params.append('er_{}_k_{}_se_{}'.format(*m.er_k_se_by_attention_index(index)))
            desc_blocks[n] = [
                f'in_channels_{in_ch}_out_channels_{out_ch}_stride_{stride}_resolution_{res}_' + param + f'_act_{use_relu}'
                for param in list_possible_params]
    return desc_blocks


def generate_latency_dict(model, file_name='lut.pkl', batch_size=1, iterations=200, target='onnx', raw=False):
    desc_blocks = extract_resolution_stride_dict(model)
    if os.path.exists(file_name):
        with open(file_name, 'rb') as f:
            dict_time = pickle.load(f)

        if raw:
            return dict_time

        for k, v in desc_blocks.items():
                desc_blocks[k] = [dict_time[i] for i in v]

        desc_blocks['general'] = dict_time['general']
        return desc_blocks

    list_conf = []
    dict_time = {}
    num_thread = torch.get_num_threads()
    torch.set_num_threads(1)
    for v in desc_blocks.values():
        for v2 in v:
            if not v2 in list_conf:
                list_conf.append(v2)

    time_other_module = measure_time_other_module(model, batch_size=batch_size, iterations=iterations, target=target)
    for b in list_conf:
        t = measure_time_inverted_bottleneck(b, batch_size=batch_size, num_iter=iterations, target=target)
        dict_time[b] = t

    dict_time['general'] = time_other_module
    torch.set_num_threads(num_thread)
    with open(file_name, 'wb') as f:
        pickle.dump(dict_time, f)

    if raw:
        return dict_time

    for k, v in desc_blocks.items():
        desc_blocks[k] = [dict_time[i] for i in v]

    desc_blocks['general'] = time_other_module

    return desc_blocks


def measure_onnx_time(model, iterations=10, input_size=224, batch_size=512):
    t = 0
    module1 = model.conv_stem
    module2 = model.blocks[0]
    module3 = model.blocks[6]
    lst_module = [module1, module2, module3]
    resolution = [224, 112, 7]
    num_channels = [3, module2[0].conv_dw.weight.shape[0], module3[0].conv.weight.shape[1]]
    if not os.path.exists('./outputs/onnx/'):
        os.mkdir('./outputs/onnx/')

    for i, (module, r, c) in enumerate(zip(lst_module, resolution, num_channels)):
        inp_ = np.random.randn(batch_size, c, r, r).astype(np.float32)
        dummy_input = torch.randn(batch_size, c, r, r).to('cpu')
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        model_onnx = f'./outputs/onnx/fixed_module_{i}.onnx'
        torch.onnx.export(model=module.cpu(), args=dummy_input,
                          f=model_onnx,
                          input_names=['input'], output_names=['output'])  #
        ort_session = ort.InferenceSession(model_onnx)
        _ = ort_session.run(None, {'input': inp_})
        total_time = 0
        for i in range(iterations):
            start = time.time()
            _ = ort_session.run(None, {'input': inp_})
            end = time.time()
            total_time += end - start

        total_time /= (iterations * batch_size)
        t += total_time

    return t


def measure_time_other_module(model, batch_size=1, iterations=200, target='cpu'):
    lst_modules = ['conv_stem', 'bn1', 'act1', 'blocks.0', 'blocks.6', 'global_pool', 'act2', 'classifier']
    if target == 'cpu':
        dict_t = measure_cpu_time(model, iterations=iterations, input_size=224, batch_size=batch_size, fp16=False,
                                  only_modules=True)
    elif target == 'onnx':
        t = measure_onnx_time(model, iterations=iterations, input_size=224, batch_size=batch_size)
        return t

    else:
        assert target == 'gpu', "target_device must be either 'cpu, 'gpu' or 'onnx'"
        dict_t = measure_gpu_time(model, iterations=iterations, input_size=224, batch_size=batch_size, fp16=False,
                                  only_modules=True)

    dict_t = dict_t[1]
    t = 0
    for e in lst_modules:
        t += dict_t[e]
    t /= batch_size
    return t


def measure_time(model, image_size=224, batch_size=1, target='cpu'):
    input = torch.rand(batch_size, 3, image_size, image_size)
    model.eval()
    device = next(model.parameters()).device
    num_thread = torch.get_num_threads()
    if target == 'cpu':
        model.cpu()
        torch.set_num_threads(1)
    else:
        model.cuda()
        input = input.cuda()
        start_gpu = torch.cuda.Event(enable_timing=True)
        end_gpu = torch.cuda.Event(enable_timing=True)
        time_all = 0

    with torch.no_grad():
        res = model(input)
        start = time.time()
        for i in range(100):
            if target != 'cpu':
                start_gpu.record()
            _ = model(input)
            if target != 'cpu':
                end_gpu.record()
                torch.cuda.synchronize()
                time_all += start_gpu.elapsed_time(end_gpu) / 1000.0

    end = time.time()
    del input, res
    gc.collect()
    model.float()
    model.to(device)
    torch.set_num_threads(num_thread)
    if target != 'cpu':
        return time_all / 100
    return (end - start) / 100


def measure_time_onnx(model):
    dummy_input = torch.randn(1, 3, 224, 224).to('cpu')
    model_onnx = f'./outputs/temp.onnx'
    torch.onnx.export(model=model.cpu(), args=dummy_input,
                      f=model_onnx,
                      input_names=['input'], output_names=['output'])  #
    print(f"model saved in {model_onnx}")

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    ort_session = ort.InferenceSession(model_onnx)
    _ = ort_session.run(None, {'input': np.random.randn(1, 3, 224, 224).astype(np.float32)})
    total_time = 0
    for i in range(1000):
        inp_ = np.random.randn(1, 3, 224, 224).astype(np.float32)
        start = time.time()
        _ = ort_session.run(None, {'input': inp_})
        end = time.time()
        total_time += end - start
    return (total_time / 1000)


def measure_time_openvino(model):
    dummy_input = torch.randn(1, 3, 224, 224).to('cpu')
    model_onnx = f'./outputs/temp.onnx'
    torch.onnx.export(model=model.cpu(), args=dummy_input,
                      f=model_onnx,
                      input_names=['input'], output_names=['output'])  #
    #TODO: Those commands are not for release
    command1 = 'python3 /mnt/workspace/ori/openvino/model-optimizer/mo.py --input_model   /mnt/workspace/yonathan/nas_niv/outputs/temp.onnx --output_dir /mnt/workspace/yonathan/nas_niv/outputs/'
    command2 = '/mnt/workspace/ori/openvino/bin-mkl/intel64/Release/benchmark_app -m /mnt/workspace/yonathan/nas_niv/outputs/temp.xml -niter 1000 -nstreams 1 -b 1'
    import subprocess
    subprocess.check_output([command1], shell=True)
    out = subprocess.check_output([command2], shell=True)
    out = out.decode("utf-8").split('\n')[-2]
    out = float(re.split('Throughput: ([0-9]+\.[0-9]+) FPS', out)[-2])
    out = 1 / out
    return out
