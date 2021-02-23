import csv
import glob
import logging
import operator
import os
import re
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch import distributed as dist

from external.distributed_manager import DistributedManager


def unwrap_model(model):
    if isinstance(model, ModelEma):
        return unwrap_model(model.ema)
    else:
        return model.module if hasattr(model, 'module') else model


def get_state_dict(model):
    return unwrap_model(model).state_dict()

def silence_PIL_warnings():
    import PIL
    wa = PIL.Image.warnings
    wa.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
    wa.filterwarnings("ignore", "(Possibly )?Palette images with Transparency", UserWarning)


class NoParsingFilter(logging.Filter):
    def filter(self, record):
        return not record.getMessage().startswith('Gradient overflow')



class FilteredPrinter(object):
    def __init__(self, filtered_print, stdout, print_bool):
        self._filtered_print = filtered_print
        self._stdout = stdout
        self._print_bool = print_bool

    def _write(self, string):
        if self._print_bool:
            self._filtered_print(string, self._stdout)

    def __getattr__(self, attr):
        if attr == 'write':
            return self._write
        return getattr(self._stdout, attr)


def filtered_print(string, stdout):
    # if not (string.startswith("Gradient overflow")):
    stdout.write(string)



class CheckpointSaver:
    def __init__(
        self,
        checkpoint_prefix='checkpoint',
        recovery_prefix='recovery',
        checkpoint_dir='',
        recovery_dir='',
        decreasing=False,
        max_history=10):

        # state
        self.checkpoint_files = []  # (filename, metric) tuples in order of decreasing betterness
        self.best_epoch = None
        self.best_metric = None
        self.curr_recovery_file = ''
        self.last_recovery_file = ''

        # config
        self.checkpoint_dir = checkpoint_dir
        self.recovery_dir = recovery_dir
        self.save_prefix = checkpoint_prefix
        self.recovery_prefix = recovery_prefix
        self.extension = '.pth.tar'
        self.decreasing = decreasing  # a lower metric is better if True
        self.cmp = operator.lt if decreasing else operator.gt  # True if lhs better than rhs
        self.max_history = max_history
        assert self.max_history >= 1

    def save_checkpoint(self, model, optimizer, args, epoch, model_ema=None, metric=None, use_amp=False):
        assert epoch >= 0
        tmp_save_path = os.path.join(self.checkpoint_dir, 'tmp' + self.extension)
        last_save_path = os.path.join(self.checkpoint_dir, 'last' + self.extension)
        self._save(tmp_save_path, model, optimizer, args, epoch, model_ema, metric, use_amp)
        if os.path.exists(last_save_path):
            os.unlink(last_save_path)  # required for Windows support.
        os.rename(tmp_save_path, last_save_path)
        worst_file = self.checkpoint_files[-1] if self.checkpoint_files else None
        if (len(self.checkpoint_files) < self.max_history
            or metric is None or self.cmp(metric, worst_file[1])):
            if len(self.checkpoint_files) >= self.max_history:
                self._cleanup_checkpoints(1)
            filename = '-'.join([self.save_prefix, str(epoch)]) + self.extension
            save_path = os.path.join(self.checkpoint_dir, filename)
            os.link(last_save_path, save_path)
            self.checkpoint_files.append((save_path, metric))
            self.checkpoint_files = sorted(
                self.checkpoint_files, key=lambda x: x[1],
                reverse=not self.decreasing)  # sort in descending order if a lower metric is not better

            checkpoints_str = "Current checkpoints:\n"
            for c in self.checkpoint_files:
                checkpoints_str += ' {}\n'.format(c)
            logging.info(checkpoints_str)

            if metric is not None and (self.best_metric is None or self.cmp(metric, self.best_metric)):
                self.best_epoch = epoch
                self.best_metric = metric
                best_save_path = os.path.join(self.checkpoint_dir, 'model_best' + self.extension)
                if os.path.exists(best_save_path):
                    os.unlink(best_save_path)
                os.link(last_save_path, best_save_path)

        return (None, None) if self.best_metric is None else (self.best_metric, self.best_epoch)

    def _save(self, save_path, model, optimizer, args, epoch, model_ema=None, metric=None, use_amp=False):
        save_state = {
            'epoch': epoch,
            'arch': args.model,
            'state_dict': get_state_dict(model),
            'optimizer': optimizer.state_dict(),
            'args': args,
            'version': 2,  # version < 2 increments epoch before save
        }
        # if use_amp and 'state_dict' in amp.__dict__:
        #     save_state['amp'] = amp.state_dict()
        if model_ema is not None:
            save_state['state_dict_ema'] = get_state_dict(model_ema)
        if metric is not None:
            save_state['metric'] = metric
        torch.save(save_state, save_path)

    def _cleanup_checkpoints(self, trim=0):
        trim = min(len(self.checkpoint_files), trim)
        delete_index = self.max_history - trim
        if delete_index <= 0 or len(self.checkpoint_files) <= delete_index:
            return
        to_delete = self.checkpoint_files[delete_index:]
        for d in to_delete:
            try:
                logging.debug("Cleaning checkpoint: {}".format(d))
                os.remove(d[0])
            except Exception as e:
                logging.error("Exception '{}' while deleting checkpoint".format(e))
        self.checkpoint_files = self.checkpoint_files[:delete_index]

    def save_recovery(self, model, optimizer, args, epoch, model_ema=None, use_amp=False, batch_idx=0):
        assert epoch >= 0
        filename = '-'.join([self.recovery_prefix, str(epoch), str(batch_idx)]) + self.extension
        save_path = os.path.join(self.recovery_dir, filename)
        self._save(save_path, model, optimizer, args, epoch, model_ema, use_amp=use_amp)
        if os.path.exists(self.last_recovery_file):
            try:
                logging.debug("Cleaning recovery: {}".format(self.last_recovery_file))
                os.remove(self.last_recovery_file)
            except Exception as e:
                logging.error("Exception '{}' while removing {}".format(e, self.last_recovery_file))
        self.last_recovery_file = self.curr_recovery_file
        self.curr_recovery_file = save_path

    def find_recovery(self):
        recovery_path = os.path.join(self.recovery_dir, self.recovery_prefix)
        files = glob.glob(recovery_path + '*' + self.extension)
        files = sorted(files)
        if len(files):
            return files[0]
        else:
            return ''


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


def get_outdir(path, *paths, inc=False):
    outdir = os.path.join(path, *paths)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    elif inc:
        count = 1
        outdir_inc = outdir + '-' + str(count)
        while os.path.exists(outdir_inc):
            count = count + 1
            outdir_inc = outdir + '-' + str(count)
            assert count < 100
        outdir = outdir_inc
        os.makedirs(outdir)
    return outdir


def update_summary(epoch, train_metrics, eval_metrics, filename, write_header=False):
    rowd = OrderedDict(epoch=epoch)
    rowd.update([('train_' + k, v) for k, v in train_metrics.items()])
    rowd.update([('eval_' + k, v) for k, v in eval_metrics.items()])
    with open(filename, mode='a') as cf:
        dw = csv.DictWriter(cf, fieldnames=rowd.keys())
        if write_header:  # first iteration (epoch == 1 can't be used)
            dw.writeheader()
        dw.writerow(rowd)




def update_tensorboard(epoch, train_metrics, eval_metrics, alpha_scalars, list_alphas, loaders, writer):
    # loaders is a dict {name:loader}
    if writer is None:
        return

    writer.add_scalars('alphas', alpha_scalars, epoch)

    for k, v in train_metrics.items():
        writer.add_scalar(f'train_{k}', v, epoch)

    for k, v in eval_metrics.items():
        writer.add_scalar(f'val_{k}', v, epoch)

    # update_alpha_beta_tensorboard(epoch, list_alphas, writer)

    for name, loader in loaders.items():
        if loader is not None:
            ims = next(iter(loader))[0]
            # take the first 8 and the first 8 of the second half. Sometime the data is split into 2...
            inds = np.concatenate((np.arange(8), np.arange(8) + len(ims) / 2))
            ims = ims[inds]
            writer.add_image(f'{name}_images',
                             torchvision.utils.make_grid(ims.data, nrow=4, normalize=True), epoch)



def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def reduce_tensor(tensor, n):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n
    return rt


def distribute_bn(model, world_size, reduce=False):
    # ensure every node has the same running bn stats
    for bn_name, bn_buf in unwrap_model(model).named_buffers(recurse=True):
        if ('running_mean' in bn_name) or ('running_var' in bn_name):
            if reduce:
                # average bn stats across whole group
                torch.distributed.all_reduce(bn_buf, op=dist.ReduceOp.SUM)
                bn_buf /= float(world_size)
            else:
                # broadcast bn stats from rank 0 to whole group
                torch.distributed.broadcast(bn_buf, 0)


class ModelEma:
    """ Model Exponential Moving Average
    Keep a moving average of everything in the model state_dict (parameters and buffers).

    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage

    A smoothed version of the weights is necessary for some training schemes to perform well.
    E.g. Google's hyper-params for training MNASNet, MobileNet-V3, EfficientNet, etc that use
    RMSprop with a short 2.4-3 epoch decay period and slow LR decay rate of .96-.99 requires EMA
    smoothing of weights to match results. Pay attention to the decay constant you are using
    relative to your update count per epoch.

    To keep EMA from using GPU resources, set device='cpu'. This will save a bit of memory but
    disable validation of the EMA weights. Validation will have to be done manually in a separate
    process, or after the training stops converging.

    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    I've tested with the sequence in my own train.py for torch.DataParallel, apex.DDP, and single-GPU.
    """

    def __init__(self, model, decay=0.9999, device='', resume=''):
        # make a copy of the model for accumulating moving average of weights
        self.ema = deepcopy(model)
        self.ema.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if device:
            self.ema.to(device=device)
        self.ema_has_module = hasattr(self.ema, 'module')
        if resume:
            self._load_checkpoint(resume)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def _load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        assert isinstance(checkpoint, dict)
        if 'state_dict_ema' in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict_ema'].items():
                # ema model may have been wrapped by DataParallel, and need module prefix
                if self.ema_has_module:
                    name = 'module.' + k if not k.startswith('module') else k
                else:
                    name = k
                new_state_dict[name] = v
            self.ema.load_state_dict(new_state_dict)
            logging.info("Loaded state_dict_ema")
        else:
            logging.warning("Failed to find state_dict_ema, starting from loaded model weights")

    def update(self, model):
        # correct a mismatch in state dict keys
        needs_module = hasattr(model, 'module') and not self.ema_has_module
        with torch.no_grad():
            msd = model.state_dict()
            for k, ema_v in self.ema.state_dict().items():
                if needs_module and k not in msd.keys():
                    k = 'module.' + k
                if k.startswith('module.') and k not in msd.keys():
                    k = k[7:]
                model_v = msd[k].detach()
                if self.device:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(ema_v * self.decay + (1. - self.decay) * model_v)



def copy_state_dict(src, target):
    with torch.no_grad():
        msd = src.state_dict()
        needs_module = hasattr(src, 'module') and not hasattr(target, 'module')
        for k, v in target.state_dict().items():
            if needs_module and k not in msd.keys():
                k = 'module.' + k
            if k.startswith('module.') and k not in msd.keys():
                k = k[7:]
            model_v = msd[k].detach()
            v.copy_(model_v)

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
    console_handler.addFilter(NoParsingFilter())


class AverageEmbeddingMeter:
    def __init__(self, centroid_shape):
        self.res = torch.zeros(centroid_shape, requires_grad=False).cuda()
        self.count = torch.zeros(len(self.res), requires_grad=False).cuda().long()

    def add(self, val, target):
        labels = target.cuda().view(target.size(0), 1).expand(-1, val.size(1))
        unique_labels, labels_count = labels.unique(dim=0, return_counts=True)
        self.res.scatter_add_(0, labels, val.float().detach())
        self.count[unique_labels[:, 0]] += labels_count

    def get(self):
        assert self.count.min() > 0, self.count
        return (self.res / self.count.float().unsqueeze(1))


def init_fc_from_centroids(model, queue, batch_size=400):
    if hasattr(model, 'module'):
        model_ = model.module
    else:
        model_ = model

    for fcc in model_.modules():
        pass
    assert isinstance(fcc, torch.nn.Linear), "Only Fully Connected is supported"

    m = fcc.weight.mean()
    sig = fcc.weight.std()

    global embedding
    embedding_dim = fcc.weight.shape[1]
    embedding = torch.zeros(batch_size, embedding_dim).cuda()
    print('embedding dim', embedding_dim)

    def extract_embed(m, i, o):
        global embedding
        if embedding.shape == i[0].shape:
            embedding[:] = i[0]
        else:
            embedding = i[0]

    h = fcc.register_forward_hook(extract_embed)
    avg = AverageEmbeddingMeter(fcc.weight.shape)
    model.eval()
    with torch.no_grad():
        import tqdm
        for data in tqdm.tqdm(queue):
            out = model(data[0].cuda())
            if embedding is not None:
                avg.add(embedding, data[1])
        new_weight = avg.get().detach()
        if DistributedManager.distributed:
            grp = DistributedManager.grp
            ws = torch.distributed.get_world_size()
            torch.distributed.all_reduce(fcc.weight.data, op=torch.distributed.ReduceOp.SUM, group=grp)
            torch.distributed.all_reduce(new_weight, op=torch.distributed.ReduceOp.SUM, group=grp)
            fcc.weight.data /= ws
            new_weight /= ws
        new_weight = nn.functional.normalize(new_weight, p=2, dim=1)
        print(f'set fc weight. mean = {m}. random sigma = {sig}, centroid sigma = {new_weight.std()}')
        fcc.weight.data.copy_(new_weight / new_weight.std() * sig)
        print(f'set fc weight. new sigma = {fcc.weight.std()}')

        fcc.bias.data[:] = 0

    model.train()
    h.remove()
