import numpy as np
import torch
from torch import nn
from nas.src.operations import AFFINE


def calculate_stride_and_padding(w_in, w_out, kernel_size, dilation):
    if w_out == 1:
        if w_in == 1 and kernel_size == 1:
            return (1, (0, 0))
        else:
            raise RuntimeError("No possible convolution of kernel size {} from input shape {} to output shape {}"
                               .format(kernel_size, w_in, w_out))

    stride = max(1, int(np.ceil(1.*(w_in - dilation * (kernel_size - 1) - 1) / (w_out - 1))))

    # Debug exception
    if stride < 1:
        raise RuntimeError(
            "stride < 1 | in: {}, kernel: {}, dilation: {}, stride: {}, out: {}"
                .format(w_in, kernel_size, dilation, stride, w_out))

    pads = ((w_out - 1) * stride + dilation * (kernel_size - 1) + 1 - w_in)

    if pads < 0:
        print("stride: {}, pads: {}".format(stride, pads))
        raise RuntimeError(
            "No possible convolution of kernel size {} and stride {} from input shape {} to output shape {}"
                .format(kernel_size, stride, w_in, w_out))

    p = int(np.floor(1.*pads / 2))
    padding = (int(np.ceil(pads)) - p, p)

    if np.floor(1.*(w_in - dilation * (kernel_size - 1) - 1 + padding[0] + padding[1] + stride) / stride) != w_out:
         raise RuntimeError(
            "Despite stride and padding calculations, output size does not match | \
            in: {}, kernel: {}, padding: {}, stride: {}, out: {}"
                .format(w_in, kernel_size, padding, stride, w_out))

    # print("in: {}, out: {}, kernel: {}, stride: {}, padding: {}".format(w_in, w_out, kernel_size, stride, padding))

    return (stride, padding)


def calc_experts_theoretical_lr(num_experts, horizon, bound, batch_size=1, eps=None):
    if bound is None or bound < 0:
        bound = 1

    effective_num_experts = num_experts
    if eps is not None:
        assert eps <= 1 and eps > 0 and num_experts > 1
        # For imbalanced attentions initialization:
        # log(N) -> log(min_unormalized_attention / total_unormalized_attention)
        effective_num_experts = (num_experts - 1) / eps

    # See Cesa chapter 2 or our slides:  eta = srt(2log(N)/T) for grad_loss in [-1,1]
    theoretical_lr = np.sqrt(2 * np.log(effective_num_experts) / horizon) / bound

    # Multiply by the batch size to allow the correction of a loss with reduction='mean' (default)
    return theoretical_lr * batch_size

def calc_initial_unnormalized_attention(num_experts, horizon, bound,
                                        lr=None, batch_size=1, grace=0, eps=None, dominant_unnormalized_attention=1):
    if bound is None or bound < 0:
        bound = 1

    if horizon is None:
        return 0, 0

    if eps is None:
        if lr is None:
            lr = calc_experts_theoretical_lr(num_experts, horizon, bound, batch_size, eps)

        normalized_attention_target = np.exp(-2 * lr * bound * (horizon - grace))
        eps = (num_experts - 1) * normalized_attention_target
    else:
        eps = (num_experts - 1) / num_experts if eps < 0 else eps
        if lr is None:
            lr = calc_experts_theoretical_lr(num_experts, horizon, bound, batch_size, eps)

    eps = np.clip(eps, 1e-5, 1-1e-5)

    initial_unnormalized_attention = eps / ((1 - eps) * (num_experts - 1)) * dominant_unnormalized_attention

    return initial_unnormalized_attention, lr


def create_layer_by_operation(op, nas_node_in, nas_node_out):
    kernel_size = op['kernel_size']
    dilation = op['dilation']
    stride_height, padding_height = calculate_stride_and_padding(nas_node_in.height, nas_node_out.height,
                                                                 kernel_size, dilation)

    stride_width, padding_width = calculate_stride_and_padding(nas_node_in.width, nas_node_out.width,
                                                               kernel_size, dilation)

    stride = (stride_height, stride_width)
    padding = padding_height + padding_width

    layer = op['fn'](nas_node_in.channels, nas_node_out.channels, stride, padding=0, affine=AFFINE, use_eca=ECA)
    if padding[0] > 0 or padding[1] > 0:
        layer = nn.Sequential(nn.ZeroPad2d(padding), layer)

    return layer


def count_params(module):
    return np.sum(np.prod(v.size()) for name, v in module.named_parameters())


def calc_wipeout_factor(lr, horizon, bound):
    return np.exp(-2 * lr * bound * horizon)

def angle(tensor1, tensor2, eps=1e-8):
    numerator = torch.sum(torch.mul(tensor1, tensor2))
    denominator = torch.norm(tensor1) * torch.norm(tensor2)

    angle = numerator / max(denominator, eps)

    return angle
