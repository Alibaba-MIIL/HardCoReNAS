""" CUDA / AMP utils
Hacked together by / Copyright 2020 Ross Wightman
"""
import torch

try:
    from apex import amp

    has_apex = True
except ImportError:
    amp = None
    has_apex = False

from external.distributed_manager import DistributedManager

import logging


class ApexScaler:
    state_dict_key = "amp"

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, step=True):
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(create_graph=create_graph)
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), clip_grad)

    def step(self, optimizer):
        optimizer.step()

    def state_dict(self):
        if 'state_dict' in amp.__dict__:
            return amp.state_dict()

    def load_state_dict(self, state_dict):
        if 'load_state_dict' in amp.__dict__:
            amp.load_state_dict(state_dict)


class NativeScaler:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, unscale=False,
                 add_opt=None, step=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if clip_grad is not None or unscale:
            self._scaler.unscale_(optimizer)
            if add_opt is not None:
                self._scaler.unscale_(add_opt)
            # unscale the gradients of optimizer's assigned params in-place
        if clip_grad is not None:
            assert parameters is not None
            torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
        skip_step = False
        for p in parameters:
            if p.grad is not None:
                if torch.any(torch.isinf(p.grad)) or torch.any(torch.isnan(p.grad)):
                    print("NaN or Inf found in gradient, skipping step")
                    skip_step = True
                    break
        if not skip_step and add_opt is not None:
            for p in add_opt.param_groups[0]['params']:
                if DistributedManager.distributed:
                    grp = DistributedManager.grp
                    ws = torch.distributed.get_world_size()
                    torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.SUM, group=grp)
                    p.grad /= ws

        if step:
            self._scaler.step(optimizer)
            self._scaler.update()

    def update(self):
        self._scaler.update()

    def step(self, optimizer):
        retval = self._scaler.step(optimizer)
        self._scaler.update()
        return retval

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)
