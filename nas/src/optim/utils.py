import torch
import numpy as np

from external.distributed_manager import DistributedManager

EPSILON = 1e-6


def flatten_attention_latency_grad_alpha_beta_blocks(list_alphas):
    '''
    Flatten all the layers attentions, measured latencies and gradients as corresponding column stack vectors
    :param list_alphas: A list of dictionaries mapping all the modules to their corresponding attentions and latencies
    :return: alpha_attention_vec: A column vector of all the attentions for all the layers
             latency_vec: A column vector of all the latencies of all the layers
             alpha_grad_vec: A column vector of all the gradients w.r.t all the attentions
             alpha_blocks: A list of lengths of the associated alphas
             beta_attention_vec: A column vector of all the beta attentions for all the sink points
             beta_grad_vec: A column vector of all the gradients w.r.t all the beta attentions
             beta_blocks: A list of lengths of the associated betas
    '''
    alpha_attention_vec = np.empty(0)
    beta_attention_vec = np.empty(0)
    latency_vec = np.empty(0)
    alpha_grad_vec = np.empty(0)
    beta_grad_vec = np.empty(0)
    alpha_blocks = []
    beta_blocks = []
    for entry in list_alphas:
        if 'alpha' in entry:
            alpha_attention_vec, latency_vec, alpha_grad_vec = \
                flatten_alpha(entry, alpha_attention_vec, latency_vec, alpha_grad_vec, alpha_blocks)

        if 'beta' in entry:
            beta_attention_vec, beta_grad_vec = \
                flatten_beta(entry, beta_attention_vec, beta_grad_vec, beta_blocks)

    return alpha_attention_vec, latency_vec, alpha_grad_vec, alpha_blocks, \
           beta_attention_vec, beta_grad_vec, beta_blocks


def flatten_alpha(entry, alpha_attention_vec, latency_vec, alpha_grad_vec, alpha_blocks):
    alpha = entry['module'].alpha.detach().clone()
    latency = entry['latency'].detach()
    alpha_grad = entry['module'].attention_grad.detach().clone() \
        if entry['module'].attention_grad is not None else None
    assert len(alpha) == len(latency)
    if alpha_grad is not None:
        assert len(alpha) == len(alpha_grad)
        if len(alpha.shape) < len(alpha_grad.shape):
            alpha_grad = torch.mean(alpha_grad, dim=1)

    alpha_blocks.append(len(alpha))
    if DistributedManager.distributed:
        grp = DistributedManager.grp
        ws = torch.distributed.get_world_size()
        torch.distributed.all_reduce(alpha, op=torch.distributed.ReduceOp.SUM, group=grp)
        alpha /= ws
        torch.distributed.all_reduce(latency, op=torch.distributed.ReduceOp.SUM, group=grp)
        latency /= ws
        if alpha_grad is not None:
            torch.distributed.all_reduce(alpha_grad, op=torch.distributed.ReduceOp.SUM, group=grp)
            alpha_grad /= ws

    alpha_attention_vec = np.concatenate((alpha_attention_vec, torch.softmax(alpha, dim=0).cpu().numpy()))
    latency_vec = np.concatenate((latency_vec, latency.squeeze().cpu().numpy()))
    if alpha_grad is not None:
        alpha_grad_vec = np.concatenate((alpha_grad_vec, alpha_grad.squeeze().cpu().numpy()))

    return alpha_attention_vec, latency_vec, alpha_grad_vec


def flatten_beta(entry, beta_attention_vec, beta_grad_vec, beta_blocks):
    beta = entry['beta'].detach()
    beta_grad = entry['module'].beta_attention_grad.detach() \
        if entry['module'].beta_attention_grad is not None else None

    if beta_grad is not None:
        assert len(beta) == len(beta_grad)
        if len(beta.shape) < len(beta_grad.shape):
            beta_grad = torch.mean(beta_grad, dim=1)

    beta_blocks.append(len(beta))
    if DistributedManager.distributed:
        grp = DistributedManager.grp
        ws = torch.distributed.get_world_size()
        torch.distributed.all_reduce(beta, op=torch.distributed.ReduceOp.SUM, group=grp)
        beta /= ws
        if beta_grad is not None:
            torch.distributed.all_reduce(beta_grad, op=torch.distributed.ReduceOp.SUM, group=grp)
            beta_grad /= ws

    beta_attention_vec = np.concatenate((beta_attention_vec, torch.softmax(beta, dim=0).cpu().numpy()))
    if beta_grad is not None:
        beta_grad_vec = np.concatenate((beta_grad_vec, beta_grad.squeeze().cpu().numpy()))

    return beta_attention_vec, beta_grad_vec


def update_attentions_inplace(list_alphas, alpha_attention_vec, beta_attention_vec=None):
    alpha_offset, beta_offset = 0, 0
    for entry in list_alphas:
        if 'alpha' in entry:
            old_logits = entry['module'].alpha
            size = len(old_logits)
            offset = alpha_offset
            attention_vec = alpha_attention_vec
            alpha_offset += size

        elif 'beta' in entry:
            old_logits = entry['beta']
            size = len(old_logits)
            offset = beta_offset
            attention_vec = beta_attention_vec
            beta_offset += size

        if attention_vec is None:
            continue

        attention = torch.from_numpy(attention_vec[offset:(offset + size)]).to(
            dtype=old_logits.dtype, device=old_logits.device).squeeze().contiguous()
        logits = calculate_attention_to_logits(attention)
        if DistributedManager.distributed:
            grp = DistributedManager.grp
            ws = torch.distributed.get_world_size()
            torch.distributed.all_reduce(logits.data, op=torch.distributed.ReduceOp.SUM, group=grp)
            logits.data /= ws

        old_logits.data.copy_(logits.data.clone())

        # # For one step optimization with aggregated gradients
        # old_grad = old_attention.grad.to(dtype=attention.dtype, device=attention.device) \
        #     if old_attention is not None and old_attention.grad is not None else None
        # attention.grad = old_grad
        # old_attention.data = attention.data


def calculate_attention_to_logits(attention):
    # For a better numerical stability
    clamped = torch.clamp_min(attention, EPSILON)
    logits = torch.log(clamped)

    return logits
