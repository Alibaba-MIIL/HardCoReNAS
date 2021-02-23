#!/usr/bin/env python
""" ImageNet Training Script

This is intended to be a lean and easily modifiable ImageNet training script that reproduces ImageNet
training results with some of the latest networks and training techniques. It favours canonical PyTorch
and standard Python style over trying to be able to 'do it all.' That said, it offers quite a few speed
and training result improvements over the usual PyTorch example scripts. Repurpose as you see fit.

This script was started from an early version of the PyTorch ImageNet example
(https://github.com/pytorch/examples/tree/master/imagenet)

NVIDIA CUDA specific speedups adopted from NVIDIA Apex examples
(https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

Hacked together by Ross Wightman (https://github.com/rwightman)
"""
import sys
import time
from contextlib import suppress
from datetime import datetime

import yaml
from torch.nn.parallel import DistributedDataParallel as NativeDDP

from external.nas_parser import *
from external.utils_pruning import build_co_train_model, compute_flops, build_KD_model
from nas.nas_utils.general_purpose import extract_structure_param_list
from timm import create_model
from timm.data import Dataset, CsvDataset, create_loader, FastCollateMixup, mixup_batch, \
    AugMixDataset, resolve_data_config
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, JsdCrossEntropy
from timm.models import resume_checkpoint, convert_splitbn_model
from timm.models.mobilenasnet import reorganize_channels, IKD_mobilenasnet_model, update_training_mode, \
    transform_model_to_mobilenet
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import *
from timm.utils_new.cuda import ApexScaler, NativeScaler


try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model

    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

torch.backends.cudnn.benchmark = True

import torchvision.utils
import gc
from tensorboardX import SummaryWriter

torch.backends.cudnn.benchmark = True

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# Dataset / Model parameters
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--csv-file', default='data.csv',
                    help='file name for csv. Expected to be in data folder')
parser.add_argument('--model', default='mobilenasnet', type=str, metavar='MODEL',
                    help='Name of model to train (default: "mobilenasnet"')
parser.add_argument('--model_IKD', default=None, type=str, metavar='MODEL',
                    help='Name of model to train ')
parser.add_argument('--use_KD', action='store_true', default=False,
                    help='Use Knowledge distillation instead of IKD')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
parser.add_argument('--initial-checkpoint_IKD', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
parser.add_argument('--no-resume-opt', action='store_true', default=False,
                    help='prevent resume of optimizer state when resuming model')
parser.add_argument('--num-classes', type=int, default=1000, metavar='N',
                    help='number of label classes (default: 1000)')
parser.add_argument('--gp', default='avg', type=str, metavar='POOL',
                    help='Type of global pool, "avg", "max", "avgmax", "avgmaxc" (default: "avg")')
parser.add_argument('--img-size', type=int, default=None, metavar='N',
                    help='Image patch size (default: None => model default)')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop percent (for validation only)')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--min-crop-factor', type=float, default=0.08,
                    help='minimum size of crop for image transformation in training')
parser.add_argument('--squish', action='store_true', default=False,
                    help='use squish for resize input image')
parser.add_argument('-b', '--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('-vb', '--validation-batch-size-multiplier', type=int, default=1, metavar='N',
                    help='ratio of validation batch size to training batch size (default: 1)')
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.)')
parser.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                    help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
parser.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                    help='Drop path rate (default: None)')
parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                    help='Drop block rate (default: None)')
parser.add_argument('--jsd', action='store_true', default=False,
                    help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
# Optimizer parameters
parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "sgd"')
parser.add_argument('--opt-eps', default=0.001, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=0.0001,
                    help='weight decay (default: 0.0001)')
# Learning rate schedule parameters
parser.add_argument('--sched', default='step', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "step"')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                    help='learning rate cycle len multiplier (default: 1.0)')
parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                    help='learning rate cycle limit')
parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='LR',
                    help='warmup learning rate (default: 0.0001)')
parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--epochs_skip', type=int, default=200, metavar='N',
                    help='number of epochs to before skip annihilation')
parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--warmup-epochs', type=int, default=3, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')
# Augmentation parameters
parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                    help='Color jitter factor (default: 0.4)')
parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5', metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". (default: None)'),
parser.add_argument('--aug-splits', type=int, default=0,
                    help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
parser.add_argument('--reprob', type=float, default=0.2, metavar='PCT',
                    help='Random erase prob (default: 0.2)')
parser.add_argument('--remode', type=str, default='pixel',
                    help='Random erase mode (default: "pixel")')
parser.add_argument('--recount', type=int, default=1,
                    help='Random erase count (default: 1)')
parser.add_argument('--resplit', action='store_true', default=False,
                    help='Do not random erase first (clean) augmentation split')
parser.add_argument('--mixup', type=float, default=0.0,
                    help='mixup alpha, mixup enabled if > 0. (default: 0.)')
parser.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                    help='turn off mixup after this epoch, disabled if 0 (default: 0)')
parser.add_argument('--smoothing', type=float, default=0.1,
                    help='label smoothing (default: 0.1)')
parser.add_argument('--train-interpolation', type=str, default='random',
                    help='Training interpolation (random, bilinear, bicubic default: "random")')
# Batch norm parameters (only works with gen_efficientnet based models currently)
parser.add_argument('--bn-tf', action='store_true', default=False,
                    help='Use Tensorflow BatchNorm defaults for models that support it (default: False)')
parser.add_argument('--bn-momentum', type=float, default=None,
                    help='BatchNorm momentum override (if not None)')
parser.add_argument('--bn-eps', type=float, default=None,
                    help='BatchNorm epsilon override (if not None)')
parser.add_argument('--sync-bn', action='store_true',
                    help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
parser.add_argument('--dist-bn', type=str, default='',
                    help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
parser.add_argument('--split-bn', action='store_true',
                    help='Enable separate BN layers per augmentation split.')
# Model Exponential Moving Average
parser.add_argument('--model-ema', type=str2bool, nargs='?', const=True, default=True,
                    help='Enable tracking moving average of model weights')
parser.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                    help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
parser.add_argument('--model-ema-decay', type=float, default=0.999,
                    help='decay factor for model weights moving average (default: 0.9998)')
# Misc
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                    help='how many batches to wait before writing recovery checkpoint')
parser.add_argument('-j', '--workers', type=int, default=16, metavar='N',
                    help='how many training processes to use (default: 16)')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--save-images', action='store_true', default=False,
                    help='save images of input bathes every log interval for debugging')
parser.add_argument('--amp', type=str2bool, nargs='?', const=True, default=True,
                    help='use NVIDIA amp for mixed precision training')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--output', default='./outputs', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')
parser.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC',
                    help='Best metric (default: "top1"')
parser.add_argument('--tta', type=int, default=0, metavar='N',
                    help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--vanishing_skip', action='store_true', default=False,
                    help='progressively annihilate the skip connection')
parser.add_argument('--drop_skip', action='store_true', default=False, help='use bernouilli droping')
parser.add_argument('--nonstrict_checkpoint', type=str2bool, nargs='?', const=True, default=True,
                    help='Ignore missmatch in size when loading model weights. Used for transfer learning')
parser.add_argument('--init_fc_from_centroids', action='store_true', default=False,
                    help='initialize fc from embedding centroids')
parser.add_argument('--no_tensorboard', action='store_true', default=False,
                    help='not write to TensorboardX.')
parser.add_argument("--single-view", action='store_true', default=False, help="train only the fc layer")
parser.add_argument("--debug", action='store_true', default=False, help="logging is set to debug")
parser.add_argument("--train_percent", type=int, default=100,
                    help="what percent of data to use for train (don't forget to leave out val")
parser.add_argument('--resnet_structure', type=int, nargs='+', default=[3, 4, 6, 3], metavar='resnetstruct',
                    help='custom resnet structure')
parser.add_argument('--resnet_block', default='Bottleneck', type=str, metavar='block',
                    help='custom resnet block')

parser.add_argument("--ema_KD", action='store_true', default=False, help="use KD from EMA")
parser.add_argument('--temperature_T', type=float, default=1,
                    help='factor for temperature of the teacher')
parser.add_argument('--temperature_S', type=float, default=1,
                    help='factor for temperature of the student')
parser.add_argument('--keep_only_correct', action='store_true', default=False,
                    help='Hard threshold for training from example')
parser.add_argument('--only_kd', action='store_true', default=False,
                    help='Use onlu kd')
parser.add_argument('--verbose', action='store_true', default=False,
                    help='Verbose mode')
parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                    help='Clip gradient norm (default: None, no clipping)')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--apex-amp', action='store_true', default=False,
                    help='Use NVIDIA Apex AMP mixed precision')
parser.add_argument('--native-amp', action='store_true', default=False,
                    help='Use Native Torch AMP mixed precision')

add_nas_to_parser(parser)

def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def get_train_val_dir(basedir):
    train_dir = val_dir = None
    for reg in 'train train_set'.split():
        if os.path.exists(os.path.join(basedir, reg)):
            train_dir = os.path.join(basedir, reg)
            break
    if train_dir is None:
        logging.error('Training folder does not exist at: {}'.format(basedir))
        exit(1)

    for reg in 'val validation val_set test'.split():
        if os.path.exists(os.path.join(basedir, reg)):
            val_dir = os.path.join(basedir, reg)
            break
    if val_dir is None:
        logging.error('Validation folder does not exist at: {}'.format(basedir))
        exit(1)

    return train_dir, val_dir


def main():
    args, args_text = _parse_args()
    default_level = logging.INFO
    if args.debug:
        default_level = logging.DEBUG

    setup_default_logging(default_level=default_level)
    args.prefetcher = not args.no_prefetcher
    args.distributed = False
    writer = None
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
        if args.distributed and args.num_gpu > 1:
            logging.warning(
                'Using more than one GPU per process in distributed mode is not allowed. Setting num_gpu to 1.')
            args.num_gpu = 1

    args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0  # global rank
    if args.distributed:
        args.num_gpu = 1
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()

    assert args.rank >= 0
    DistributedManager.set_args(args)
    sys.stdout = FilteredPrinter(filtered_print, sys.stdout, args.rank == 0)
    if args.distributed:
        logging.info('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
                     % (args.rank, args.world_size))

    else:
        logging.info('Training with a single process on %d GPUs.' % args.num_gpu)

    torch.manual_seed(args.seed + args.rank)

    if not args.no_tensorboard and DistributedManager.is_master():
        writer = SummaryWriter('outputs')

    if os.path.exists(os.path.join(args.data, args.csv_file)):
        dataset_train = CsvDataset(os.path.join(args.data, args.csv_file),
                                   single_view=args.single_view, data_percent=args.train_percent)
        dataset_eval = CsvDataset(os.path.join(args.data, args.csv_file),
                                  single_view=True, data_percent=10, reverse_order=True)
    else:
        train_dir, eval_dir = get_train_val_dir(args.data)
        dataset_train = Dataset(train_dir)
        if args.train_percent < 100:
            dataset_train, dataset_valid = dataset_train.split_dataset(
                1.0 * args.train_percent / 100.0)
        if args.train_elastic_model:
            dataset_extract_channel, _ = dataset_train.split_dataset(
                1.0 * 10 / 100.0)

        dataset_eval = Dataset(eval_dir)
    logging.info(f'Training data has {len(dataset_train)} images')
    args.num_classes = len(dataset_train.class_to_idx)
    logging.info(f'setting num classes to {args.num_classes}')

    model_IKD = None

    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_connect_rate=args.drop_connect,
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        bn_tf=args.bn_tf,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        checkpoint_path=args.initial_checkpoint if not args.train_elastic_model else args.initial_checkpoint_IKD,
        strict=not args.nonstrict_checkpoint,
        resnet_structure=args.resnet_structure,
        resnet_block=args.resnet_block,
        heaviest_network=args.heaviest_network or args.train_elastic_model,
        use_kernel_3=args.use_kernel_3,
        exp_r=args.exp_r,
        depth=args.depth,
        reduced_exp_ratio=args.reduced_exp_ratio,
        use_dedicated_pwl_se=args.use_dedicated_pwl_se,
        force_sync_gpu=args.force_sync_gpu,
        multipath_sampling=args.multipath_sampling,
        mobilenet_string=args.mobilenet_string if not args.transform_model_to_mobilenet else '',
        no_swish=args.no_swish,
        search_mode=False,
        use_swish=args.use_swish
    )
    if args.force_se and 'mobilenasnet' in args.model:
        model.set_force_se(True)

    list_alphas = None
    if 'mobilenasnet' in args.model and args.transform_model_to_mobilenet:
        if args.set_alpha_beta:
            print(
                f"Setting model to er={args.mobilenasnet_er}, k={args.mobilenasnet_kernel}, depth={args.mobilenasnet_depth}")
            model.set_all_alpha(er=args.mobilenasnet_er, k=args.mobilenasnet_kernel, se=0.25)
            model.set_all_beta(args.mobilenasnet_depth)

        model2, string_model = transform_model_to_mobilenet(model, load_weight=not args.no_weight_loading,
                                                            mobilenet_string=args.mobilenet_string)
        del model
        model = model2
        model.eval()
        print("Converting model to MobileNet_v3 space")

    if args.train_elastic_model:
        # TODO: edit here
        assert args.initial_checkpoint is not None
        assert 'mobilenasnet' in args.model
        assert args.initial_checkpoint_IKD != ''
        args.initial_checkpoint_IKD = ''
        if args.initial_checkpoint == '':
            reorganize_channels(model, dataset_extract_channel, args)

        model2 = create_model(
            args.model,
            pretrained=args.pretrained,
            num_classes=args.num_classes,
            drop_rate=args.drop,
            drop_connect_rate=args.drop_connect,
            drop_path_rate=args.drop_path,
            drop_block_rate=args.drop_block,
            global_pool=args.gp,
            bn_tf=args.bn_tf,
            bn_momentum=args.bn_momentum,
            bn_eps=args.bn_eps,
            checkpoint_path=args.initial_checkpoint,
            strict=not args.nonstrict_checkpoint,
            resnet_structure=args.resnet_structure,
            resnet_block=args.resnet_block,
            heaviest_network=False,
            use_kernel_3=args.use_kernel_3,
            exp_r=args.exp_r,
            depth=args.depth,
            reduced_exp_ratio=args.reduced_exp_ratio,
            use_dedicated_pwl_se=args.use_dedicated_pwl_se,
            force_sync_gpu=args.force_sync_gpu,
            multipath_sampling=args.multipath_sampling,
            no_swish=args.no_swish,
            search_mode=False,
            use_swish=args.use_swish
        )
        model2.set_hard_backprop(True)
        model_IKD = IKD_mobilenasnet_model(model, model2, args.gamma_knowledge, args.ikd_dividor, args.real_KD)
        if args.initial_checkpoint == '':
            model2.load_from_super_network(model)
            model2.set_last_alpha()
            model2.set_last_beta()
            with torch.no_grad():
                x = torch.rand(64, 3, 224, 224).cuda()
                model_IKD.train().cuda()
                model_IKD.module1.eval()
                model_IKD.module2.eval()
                out, loss = model_IKD(x)
            assert torch.abs(loss) == 0 or (args.real_KD and torch.abs(loss) < 1e-4)
            del loss, x, out
            gc.collect()
            torch.cuda.empty_cache()

        else:
            model2.set_last_alpha()
            model2.set_last_beta()

        model2.set_hard_backprop(args.hard_backprop)
        model = model_IKD
        model_IKD = None
        model.train().cpu()
        if args.train_elastic_model:
            update_training_mode(model.module2, args.mode_training,
                                 prefer_higher_width_fact=args.prefer_higher_width_fact,
                                 prefer_higher_k_fact=args.prefer_higher_k_fact,
                                 prefer_higher_beta_fact=args.prefer_higher_beta_fact)

    if args.local_rank == 0:
        logging.info('Model %s created, param count: %d' %
                     (args.model, sum([m.numel() for m in model.parameters()])))

    data_config = resolve_data_config(vars(args), model=model, verbose=False)
    model.eval()
    flops = compute_flops(model, data_config['input_size'])

    if args.local_rank == 0:
        logging.info('Model %s flops: %f GFlops' %
                     (args.model, flops / 1e9))

    if args.model_IKD is not None:
        args.smoothing = 0
        if DistributedManager.is_master():
            logging.info("Using KD or IKD, label smoothing is set to zero ")

        model_IKD = create_model(
            args.model_IKD,
            pretrained=args.pretrained,
            num_classes=args.num_classes,
            drop_rate=args.drop,
            drop_connect_rate=args.drop_connect,
            drop_path_rate=args.drop_path,
            drop_block_rate=args.drop_block,
            global_pool=args.gp,
            bn_tf=args.bn_tf,
            bn_momentum=args.bn_momentum,
            bn_eps=args.bn_eps,
            checkpoint_path=args.initial_checkpoint_IKD,
            resnet_structure=args.resnet_structure,
            resnet_block=args.resnet_block,
            heaviest_network=True,
            no_swish=args.no_swish,
            search_mode=False
        )
        model_IKD.eval()
        model_IKD.requires_grad = False
        for n, m in model_IKD.named_parameters():
            m.requires_grad = False
        if args.local_rank == 0:
            logging.info('Model IKD to train %s created, param count: %d' %
                         (args.model_IKD, sum([m.numel() for m in model.parameters()])))

        flops = compute_flops(model_IKD, data_config['input_size'])

        if args.local_rank == 0:
            logging.info('Model IKD %s flops: %f GFlops' %
                         (args.model_IKD, flops / 1e9))

    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, 'A split of 1 makes no sense'
        num_aug_splits = args.aug_splits

    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        model = convert_splitbn_model(model, max(num_aug_splits, 2))

    use_amp = None
    if args.amp:
        # for backwards compat, `--amp` arg tries apex before native amp
        if has_apex:
            args.apex_amp = True

        elif has_native_amp:
            args.native_amp = True

    if args.apex_amp and has_apex:
        use_amp = 'apex'

    elif args.native_amp and has_native_amp:
        use_amp = 'native'

    elif args.apex_amp or args.native_amp:
        logging.warning("Neither APEX or native Torch AMP is available, using float32. "
                        "Install NVIDA apex or upgrade to PyTorch 1.6")

    if args.num_gpu > 1:
        if use_amp == 'apex':
            logging.warning(
                'Apex AMP does not work well with nn.DataParallel, disabling. Use DDP or Torch AMP.')
            use_amp = None

        model = nn.DataParallel(model, device_ids=list(range(args.num_gpu))).cuda()
        if model_IKD is not None:
            model_IKD = nn.DataParallel(model_IKD, device_ids=list(range(args.num_gpu))).cuda()

        assert not args.channels_last, "Channels last not supported with DP, use DDP."

    else:
        model.cuda()
        model.train()
        if model_IKD is not None:
            model_IKD = model_IKD.cuda()
            model_IKD.eval()

        if args.channels_last:
            model = model.to(memory_format=torch.channels_last)
            if model_IKD is not None:
                model_IKD = model_IKD.to(memory_format=torch.channels_last)

    if model_IKD is not None:
        if args.use_KD:
            model_new = build_KD_model(model_IKD, model.module.cpu() if hasattr(model, 'module') else model.cpu(),
                                       alpha_KD=args.gamma_knowledge)

        else:
            model_new = build_co_train_model(model_IKD, model.module.cpu() if hasattr(model, 'module') else model.cpu(),
                                             gamma=args.gamma_knowledge)

        del model_IKD
        del model
        gc.collect()
        model = model_new
        model = model.cuda()

    model.cuda()
    model.train()
    if args.train_elastic_model:
        update_training_mode(model.module2, args.mode_training, prefer_higher_width_fact=args.prefer_higher_width_fact,
                             prefer_higher_k_fact=args.prefer_higher_k_fact,
                             prefer_higher_beta_fact=args.prefer_higher_beta_fact)

    optimizer = create_optimizer(args, model)

    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if use_amp == 'apex':
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        loss_scaler = ApexScaler()
        if args.local_rank == 0:
            logging.info('Using NVIDIA APEX AMP. Training in mixed precision.')

    elif use_amp == 'native':
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        if args.local_rank == 0:
            logging.info('Using native Torch AMP. Training in mixed precision.')

    else:
        if args.local_rank == 0:
            logging.info('AMP not enabled. Training in float32.')

    # optionally resume from a checkpoint
    resume_state = {}
    resume_epoch = None
    if args.resume:
        resume_state, resume_epoch = resume_checkpoint(model, args.resume)

    if resume_state and not args.no_resume_opt:
        if 'optimizer' in resume_state:
            if args.local_rank == 0:
                logging.info('Restoring Optimizer state from checkpoint')
            optimizer.load_state_dict(resume_state['optimizer'])

        if use_amp and 'amp' in resume_state and 'load_state_dict' in amp.__dict__:
            if args.local_rank == 0:
                logging.info('Restoring NVIDIA AMP state from checkpoint')
            amp.load_state_dict(resume_state['amp'])

    del resume_state

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume=args.resume)

    if args.distributed:
        if args.sync_bn:
            assert not args.split_bn
            try:
                if has_apex:
                    model = convert_syncbn_model(model)

                else:
                    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

                if args.local_rank == 0:
                    logging.info(
                        'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                        'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')

            except Exception as e:
                logging.error('Failed to enable Synchronized BatchNorm. Install Apex or Torch >= 1.1')
        if has_apex and use_amp != 'native':
            # Apex DDP preferred unless native amp is activated
            if args.local_rank == 0:
                logging.info("Using NVIDIA APEX DistributedDataParallel.")

            model = ApexDDP(model, delay_allreduce=True)

        else:
            if args.local_rank == 0:
                logging.info("Using native Torch DistributedDataParallel.")
            model = NativeDDP(model, device_ids=[args.local_rank],
                              find_unused_parameters=True)  # can use device str in Torch >= 1.1

            # NOTE: EMA model does not need to be wrapped by DDP

    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch

    elif resume_epoch is not None:
        start_epoch = resume_epoch

    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    if args.local_rank == 0:
        logging.info('Scheduled epochs: {}'.format(num_epochs))

    collate_fn = None
    if args.prefetcher and args.mixup > 0:
        assert not num_aug_splits  # collate conflict (need to support deinterleaving in collate mixup)
        collate_fn = FastCollateMixup(args.mixup, args.smoothing, args.num_classes)

    if num_aug_splits > 1:
        dataset_train = AugMixDataset(dataset_train, num_splits=num_aug_splits)

    loader_train = create_loader(
        dataset_train,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        is_training=True,
        use_prefetcher=args.prefetcher,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        re_split=args.resplit,
        color_jitter=args.color_jitter,
        min_crop_factor=args.min_crop_factor,
        auto_augment=args.aa,
        num_aug_splits=num_aug_splits,
        interpolation=args.train_interpolation,
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        collate_fn=collate_fn,
        pin_memory=args.pin_mem,
        squish=args.squish
    )
    loader_valid = None
    if args.train_percent < 100:
        loader_valid = create_loader(
            dataset_valid,
            input_size=data_config['input_size'],
            batch_size=args.batch_size,
            is_training=False,
            use_prefetcher=args.prefetcher,
            interpolation=data_config['interpolation'],
            mean=data_config['mean'],
            std=data_config['std'],
            num_workers=args.workers,
            distributed=args.distributed,
            collate_fn=collate_fn,
            crop_pct=data_config['crop_pct'],
            pin_memory=args.pin_mem,
            squish=args.squish,
            infinite_loader=True,
            force_data_sampler=True
        )

    loader_eval = create_loader(
        dataset_eval,
        input_size=data_config['input_size'],
        batch_size=args.validation_batch_size_multiplier * args.batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        crop_pct=data_config['crop_pct'],
        pin_memory=args.pin_mem,
        squish=args.squish,
    )

    if args.jsd:
        assert num_aug_splits > 1  # JSD only valid with aug splits set
        train_loss_fn = JsdCrossEntropy(num_splits=num_aug_splits, smoothing=args.smoothing).cuda()
        validate_loss_fn = nn.CrossEntropyLoss().cuda()

    elif args.mixup > 0.:
        # smoothing is handled with mixup label transform
        train_loss_fn = SoftTargetCrossEntropy().cuda()
        validate_loss_fn = nn.CrossEntropyLoss().cuda()

    elif args.smoothing:
        train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing).cuda()
        validate_loss_fn = nn.CrossEntropyLoss().cuda()

    else:
        train_loss_fn = nn.CrossEntropyLoss().cuda()
        validate_loss_fn = train_loss_fn

    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    saver = None
    output_dir = ''
    if args.local_rank == 0:
        output_base = args.output if args.output else './output'
        exp_name = '-'.join([
            datetime.now().strftime("%Y%m%d-%H%M%S"),
            args.model,
            str(data_config['input_size'][-1])
        ])
        output_dir = get_outdir(output_base, 'train', exp_name)
        decreasing = True if eval_metric == 'loss' else False
        saver = CheckpointSaver(checkpoint_dir=output_dir, decreasing=decreasing)
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)

    if args.init_fc_from_centroids:
        init_fc_from_centroids(model, loader_train)

    try:
        step_counter = 0
        for epoch in range(start_epoch, num_epochs):
            torch.cuda.empty_cache()
            model.train()
            if args.train_elastic_model:
                update_training_mode(model.module.module2, args.mode_training,
                                     prefer_higher_width_fact=args.prefer_higher_width_fact,
                                     prefer_higher_k_fact=args.prefer_higher_k_fact,
                                     prefer_higher_beta_fact=args.prefer_higher_beta_fact)
                update_training_mode(model_ema.ema.module2, args.mode_training,
                                     prefer_higher_width_fact=args.prefer_higher_width_fact,
                                     prefer_higher_k_fact=args.prefer_higher_k_fact,
                                     prefer_higher_beta_fact=args.prefer_higher_beta_fact)
                model_ema.ema.module2.requires_grad_(False)
                model_ema.ema.module2.eval()

            if args.distributed:
                loader_train.sampler.set_epoch(epoch)

            train_metrics, step_counter = train_epoch(
                epoch, model, loader_train, loader_valid, optimizer, train_loss_fn, args,
                lr_scheduler=lr_scheduler, saver=saver, output_dir=output_dir,
                amp_autocast=amp_autocast, loss_scaler=loss_scaler, model_ema=model_ema,
                step_counter=step_counter)
            if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                if args.local_rank == 0:
                    logging.info("Distributing BatchNorm running means and vars")

                distribute_bn(model, args.world_size, args.dist_bn == 'reduce')

            torch.cuda.empty_cache()
            gc.collect()
            eval_metrics = validate(model, loader_eval, validate_loss_fn, args)
            if model_ema is not None and not args.model_ema_force_cpu:
                if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                    distribute_bn(model_ema, args.world_size, args.dist_bn == 'reduce')

                ema_eval_metrics = validate(
                    model_ema.ema, loader_eval, validate_loss_fn, args, log_suffix=' (EMA)')
                eval_metrics = ema_eval_metrics

            if lr_scheduler is not None:
                # step LR for next epoch
                lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

            update_summary(
                epoch, train_metrics, eval_metrics, os.path.join(output_dir, 'summary.csv'),
                write_header=best_metric is None)
            if not args.no_tensorboard and DistributedManager.is_master():
                loaders = dict(train=loader_train, val=loader_eval)
                update_tensorboard(epoch, train_metrics, eval_metrics, list_alphas, loaders, writer)

            if saver is not None:
                # save proper checkpoint with eval metric
                save_metric = eval_metrics[eval_metric]
                best_metric, best_epoch = saver.save_checkpoint(
                    model, optimizer, args,
                    epoch=epoch, model_ema=model_ema, metric=save_metric, use_amp=use_amp)

    except KeyboardInterrupt:
        pass

    if best_metric is not None:
        logging.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))


def train_epoch(epoch, model, loader, loader_valid, optimizer, loss_fn, args,
                lr_scheduler=None, saver=None, output_dir='', amp_autocast=suppress, loss_scaler=None, model_ema=None,
                step_counter=0):
    if args.prefetcher and args.mixup > 0 and loader.mixup_enabled:
        if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
            loader.mixup_enabled = False

    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)

    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        if not args.prefetcher:
            input, target = input.cuda(), target.cuda()
            if args.mixup > 0.:
                input, target = mixup_batch(
                    input, target,
                    alpha=args.mixup, num_classes=args.num_classes, smoothing=args.smoothing,
                    disable=args.mixup_off_epoch and epoch >= args.mixup_off_epoch)

        with amp_autocast():
            if args.use_KD:
                out = model(input, target)

            else:
                out = model(input)

            if isinstance(out, (tuple, list)):
                output, loss2 = out
                if args.only_kd:
                    loss = loss2

                else:
                    loss = loss_fn(output, target) + loss2

            elif not args.train_elastic_model:
                output = out
                loss = loss_fn(output, target)

            else:
                loss = out

        if not args.distributed:
            losses_m.update(loss.item(), input.size(0))

        # Add speed penalties
        # # Debug
        # dataset = 'train' if loader_valid is None or loader_valid != loader else 'val'

        optimizer.zero_grad()
        if loss_scaler is not None:
            # optimizer.step() happens inside loss_scaler()
            loss_scaler(
                loss, optimizer, clip_grad=args.clip_grad, parameters=model.parameters(), create_graph=second_order)

        else:
            loss.backward(create_graph=second_order)
            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

            optimizer.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        num_updates += 1
        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group['lr'] if 'lr' in param_group.keys() else 0 for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)
            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                losses_m.update(reduced_loss.item(), input.size(0))

            if args.local_rank == 0:
                logging.info(
                    'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                    'Loss: {loss.val:>9.6f} ({loss.avg:>6.4f})  '
                    'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                    '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    'LR: {lr:.3e}  '
                    'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        epoch,
                        batch_idx, len(loader),
                        100. * batch_idx / last_idx,
                        loss=losses_m,
                        batch_time=batch_time_m,
                        rate=input.size(0) * args.world_size / batch_time_m.val,
                        rate_avg=input.size(0) * args.world_size / batch_time_m.avg,
                        lr=lr,
                        data_time=data_time_m))

                if args.save_images and output_dir:
                    torchvision.utils.save_image(
                        input,
                        os.path.join(output_dir, 'train-batch-%d.jpg' % batch_idx),
                        padding=0,
                        normalize=True)

        if saver is not None and args.recovery_interval and (
            last_batch or (batch_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(
                model, optimizer, args, epoch, model_ema=model_ema, use_amp=False, batch_idx=batch_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()
        if args.debug:
            break

        step_counter += 1

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    return OrderedDict([('loss', losses_m.avg)]), step_counter


def validate(model, loader, loss_fn, args, log_suffix=''):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx

            if not args.prefetcher:
                input = input.cuda()
                target = target.cuda()

            output = model(input)
            if isinstance(output, (tuple, list)):
                output = output[0]

            # augmentation reduction
            reduce_factor = args.tta
            if reduce_factor > 1:
                output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0:target.size(0):reduce_factor]

            loss = loss_fn(output, target)
            k = min(5, args.num_classes)
            acc1, acc5 = accuracy(output, target, topk=(1, k))

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)

            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if args.local_rank == 0 and (last_batch or batch_idx % args.log_interval == 0):
                log_name = 'Test' + log_suffix
                logging.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                    'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                        log_name, batch_idx, last_idx, batch_time=batch_time_m,
                        loss=losses_m, top1=top1_m, top5=top5_m))

    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])

    return metrics


if __name__ == '__main__':
    main()
