#!/usr/bin/env python
import sys
import time
from contextlib import suppress
from datetime import datetime

import yaml
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from tqdm import tqdm

from external.nas_parser import *
from nas.nas_utils.general_purpose import extract_structure_param_list, target_time_loss, \
    freeze_weights_unfreeze_alphas, get_stage_block_from_name, STAGE_BLOCK_DELIMITER, OptimLike, \
    update_alpha_beta_tensorboard
from nas.src.optim.block_frank_wolfe import flatten_attention_latency_grad_alpha_beta_blocks
from timm import create_model
from timm.data import Dataset, CsvDataset, create_loader, FastCollateMixup, resolve_data_config
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, JsdCrossEntropy
from timm.models import resume_checkpoint, convert_splitbn_model
from timm.models.mobilenasnet import transform_model_to_mobilenet, measure_time
from timm.optim import create_optimizer_alpha
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

import gc
from tensorboardX import SummaryWriter

torch.backends.cudnn.benchmark = True

np.set_printoptions(threshold=sys.maxsize, suppress=True, precision=6)

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
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
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
parser.add_argument('-b', '--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 16)')
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
parser.add_argument('--nonstrict_checkpoint', type=str2bool, nargs='?', const=True, default=True,
                    help='Ignore missmatch in size when loading model weights. Used for transfer learning')
parser.add_argument('--tensorboard', action='store_true', default=False,
                    help='Write to TensorboardX')
parser.add_argument("--single-view", action='store_true', default=False,
                    help="train only the fc layer")
parser.add_argument("--debug", action='store_true', default=False,
                    help="logging is set to debug")
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
                    help='Hard threshold for training from example')
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

    if args.tensorboard and DistributedManager.is_master():
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

        dataset_eval = Dataset(eval_dir)

    logging.info(f'Training data has {len(dataset_train)} images')
    args.num_classes = len(dataset_train.class_to_idx)
    logging.info(f'setting num classes to {args.num_classes}')

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
        checkpoint_path=args.initial_checkpoint,
        strict=not args.nonstrict_checkpoint,
        resnet_structure=args.resnet_structure,
        resnet_block=args.resnet_block,
        heaviest_network=args.heaviest_network,
        use_kernel_3=args.use_kernel_3,
        exp_r=args.exp_r,
        depth=args.depth,
        reduced_exp_ratio=args.reduced_exp_ratio,
        use_dedicated_pwl_se=args.use_dedicated_pwl_se,
        force_sync_gpu=args.force_sync_gpu,
        multipath_sampling=args.multipath_sampling,
        use_softmax=args.use_softmax,
        detach_gs=args.detach_gs,
        no_swish=args.no_swish,
        search_mode=True
    )
    if args.force_se and 'mobilenasnet' in args.model:
        model.set_force_se(True)

    list_alphas = None

    if args.qc_init:
        if args.init_to_biggest_alpha:
            model.set_all_alpha(er=6, k=5, se=0.25 if args.force_se else 0, use_only=False)
        else:
            model.set_all_alpha(er=3, k=3, se=0.25 if args.force_se else 0, use_only=False)

        model.set_all_beta(2, use_only=False)

    elif args.init_to_smallest:
        model.set_all_alpha(er=3, k=3, se=0.25 if args.force_se else 0, use_only=False)
        model.set_all_beta(2, use_only=False)

    elif args.init_to_biggest:
        model.set_last_alpha(use_only=False)
        model.set_last_beta(use_only=False)

    elif args.init_to_biggest_alpha:
        model.set_all_alpha(er=6, k=5, se=0.25 if args.force_se else 0, use_only=False)
        model.set_all_beta(2, use_only=False)

    else:
        model.set_uniform_alpha()
        model.set_uniform_beta(stage=1)

    if args.local_rank == 0:
        logging.info('Model %s created, param count: %d' %
                     (args.model, sum([m.numel() for m in model.parameters()])))

    data_config = resolve_data_config(vars(args), model=model, verbose=False)
    model.eval()

    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, 'A split of 1 makes no sense'
        num_aug_splits = args.aug_splits

    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        model = convert_splitbn_model(model, max(num_aug_splits, 2))

    use_amp = None
    if args.amp:
        # For backwards compat, `--amp` arg tries apex before native amp
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

        assert not args.channels_last, "Channels last not supported with DP, use DDP."

    else:
        model.cuda()
        model.train()
        if args.channels_last:
            model = model.to(memory_format=torch.channels_last)

    model.cuda()
    model.train()

    optim = None
    list_alphas = None
    fixed_latency = 0
    if args.search_elastic_model:
        model.set_hard_backprop(False)
        model.eval()
        with torch.no_grad():
            x = torch.rand(64, 3, 224, 224).cuda()
            out = model(x)
        del out, x
        gc.collect()
        torch.cuda.empty_cache()
        list_alphas, fixed_latency = extract_structure_param_list(model, file_name=args.lut_filename,
                                                              batch_size=args.lut_measure_batch_size,
                                                              repeat_measure=args.repeat_measure,
                                                              target_device=args.target_device)

    fixed_latency = args.latency_corrective_slope * fixed_latency + args.latency_corrective_intercept

    optim2 = None
    if args.train_nas or args.search_elastic_model and not args.fixed_alpha:
        optim = create_optimizer_alpha(args, list_alphas, args.lr_alphas)
        if hasattr(optim, 'fixed_latency'):
            optim.fixed_latency = fixed_latency

        if args.nas_optimizer.lower() == 'sgd':
            args2 = deepcopy(args)
            args2.nas_optimizer = 'block_frank_wolfe'
            optim2 = create_optimizer_alpha(args2, list_alphas, args.lr_alphas)
            optim2.fixed_latency = fixed_latency

    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if use_amp == 'apex':
        if optim is not None:
            model, optim = amp.initialize(model, optim, opt_level='O1')

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
        if use_amp and 'amp' in resume_state and 'load_state_dict' in amp.__dict__:
            if args.local_rank == 0:
                logging.info('Restoring NVIDIA AMP state from checkpoint')

            amp.load_state_dict(resume_state['amp'])

    del resume_state

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

            # NOTE: EMA model does not need to be wrapped by DDP
            model = NativeDDP(model, device_ids=[args.local_rank], find_unused_parameters=True)

    collate_fn = None
    if args.prefetcher and args.mixup > 0:
        assert not num_aug_splits  # collate conflict (need to support deinterleaving in collate mixup)
        collate_fn = FastCollateMixup(args.mixup, args.smoothing, args.num_classes)

    dataset_val = dataset_valid if args.train_percent < 100 else dataset_eval
    loader_valid = create_loader(
        dataset_val,
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

    # Disable weights gradients and BN statistics and enable alpha-beta gradients
    freeze_weights_unfreeze_alphas(model, optim)

    alpha_attention_vec, _, alpha_grad_vec, alpha_blocks, beta_attention_vec, beta_grad_vec, beta_blocks = \
        flatten_attention_latency_grad_alpha_beta_blocks(list_alphas)

    print('alpha_attention_vec')
    print(np.reshape(alpha_attention_vec, (len(alpha_blocks), -1)))

    print('beta_attention_vec')
    print(np.reshape(beta_attention_vec, (len(beta_blocks), -1)))
    interrupted = False

    try:
        loader_valid = iter(loader_valid)
        torch.cuda.empty_cache()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        if args.qc_init:
            print('QC balanced Init:')
            if optim2 is not None:
                optim2.bc_qp_init()
            else:
                optim.bc_qp_init()

            alpha_attention_vec, _, _, alpha_blocks, beta_attention_vec, _, beta_blocks = \
                                            flatten_attention_latency_grad_alpha_beta_blocks(list_alphas)
            check_rounding_constraint(optim2 if optim2 is not None else optim,
                                      alpha_attention_vec, beta_attention_vec, alpha_blocks, beta_blocks)

        epoch = 0
        _ = optim.set_epoch(0) if hasattr(optim, 'set_epoch') else None
        _ = optim.set_writer(writer) if hasattr(optim, 'set_writer') else None

        if not 'frank_wolfe' in args.nas_optimizer:
            update_alpha_beta_tensorboard(0, list_alphas, writer)

        bar = tqdm(range(args.bcfw_steps)) if args.local_rank == 0 else range(args.bcfw_steps)
        gpu_h_agg = torch.zeros(1).cuda()
        gpu_h_fw = torch.zeros(1).cuda()
        for k in bar:
            model.temperature = calculate_temperature(k, T0=args.init_temperature, Tf=args.final_temperature,
                                                      tf=args.temperature_annealing_period * args.bcfw_steps,
                                                      policy=args.annealing_policy)
            if writer is not None:
                writer.add_scalar('Temperature', model.temperature, k)

            if args.aggregate_grads_steps is not None:
                start.record()
                compute_and_update_list_alphas(list_alphas, local_rank=args.local_rank,
                                               steps=args.aggregate_grads_steps, model=model, loss_fn=validate_loss_fn,
                                               loader=loader_valid, optimizer=optim,
                                               loss_scaler=loss_scaler, amp_autocast=amp_autocast,
                                               prefetcher=args.prefetcher, writer=writer,
                                               inference_time_limit=args.inference_time_limit,
                                               target_time_constraint=args.target_time_constraint)
                end.record()
                torch.cuda.synchronize()
                gpu_h_agg += start.elapsed_time(end) / 1e3 / 60 / 60

            if not 'frank_wolfe' in args.nas_optimizer:
                loss_time = target_time_loss(list_alphas, args.inference_time_limit - fixed_latency)
                if DistributedManager.is_master():
                    writer.add_scalar('loss_target_time', loss_time, k)
                print(f"TF-NAS loss is {loss_time}")
                loss_time = args.target_time_constraint * loss_time
                if loss_scaler is not None:
                    loss_scaler(
                        loss_time, optim, parameters=model.parameters(), unscale=False, step=False)
                else:
                    loss_time.backward()

            for _ in range(args.steps_per_grad):
                epoch += 1
                _ = optim.set_epoch(epoch) if hasattr(optim, 'set_epoch') else None
                start.record()
                _ = optim.step() if loss_scaler is None else loss_scaler.step(optim)
                end.record()
                torch.cuda.synchronize()
                gpu_h_fw += start.elapsed_time(end) / 1e3 / 60 / 60
                if not 'frank_wolfe' in args.nas_optimizer:
                    update_alpha_beta_tensorboard(k, list_alphas, writer)

    except KeyboardInterrupt:
        interrupted = True
        pass

    print_solution(list_alphas, optim, args)
    if saver is not None:
        saver.save_checkpoint(model, optim, args, epoch=k, metric=0)

    try:
        if not args.fine_tune_alpha or interrupted:
            raise KeyboardInterrupt()

        # Set beta to argmax
        model.set_argmax_alpha_beta(only_beta=True, use_only=False) if hasattr(model, 'set_argmax_alpha_beta') \
            else model.module.set_argmax_alpha_beta(only_beta=True, use_only=False)

        optim.only_alpha = True
        optim.reset_gamma_step()
        bar = tqdm(range(args.bcfw_steps)) if args.local_rank == 0 else range(args.bcfw_steps)
        for k in bar:
            model.temperature = calculate_temperature(k, T0=args.init_temperature, Tf=args.final_temperature,
                                                      tf=args.temperature_annealing_period * args.bcfw_steps,
                                                      policy=args.annealing_policy)
            if writer is not None:
                writer.add_scalar('Temperature', model.temperature, args.bcfw_steps + k)

            optim.set_epoch(args.bcfw_steps + k)
            if args.aggregate_grads_steps is not None:
                start.record()
                compute_and_update_list_alphas(list_alphas, local_rank=args.local_rank,
                                               steps=args.aggregate_grads_steps, model=model, loss_fn=validate_loss_fn,
                                               loader=loader_valid, optimizer=optim,
                                               loss_scaler=loss_scaler, amp_autocast=amp_autocast,
                                               prefetcher=args.prefetcher)
                end.record()
                torch.cuda.synchronize()
                gpu_h_agg += start.elapsed_time(end) / 1e3 / 60 / 60

            epoch += 1
            _ = optim.set_epoch(epoch) if hasattr(optim, 'set_epoch') else None
            start.record()
            _ = optim.step() if loss_scaler is None else loss_scaler.step(optim)
            end.record()
            torch.cuda.synchronize()
            gpu_h_fw += start.elapsed_time(end) / 1e3 / 60 / 60

        print_solution(list_alphas, optim, args)
    except KeyboardInterrupt:
        pass

    # No temperature from now on
    model.temperature = 1

    print('------------------------Sparsify --------------------------------')
    if not isinstance(optim, torch.optim.SGD):
        optim.sparsify()
    else:
        optim2.sparsify()
    print_solution(list_alphas, optim, args)
    if saver is not None:
        saver.save_checkpoint(model, optim, args, epoch=args.bcfw_steps + k, metric=1)

    print('----------------------- argmax --------------------------')
    # Set alpha and beta to argmax
    model.set_argmax_alpha_beta() if hasattr(model, 'set_argmax_alpha_beta') else model.module.set_argmax_alpha_beta()

    if saver is not None:
        saver.save_checkpoint(model, optim, args, epoch=args.bcfw_steps + k + 1, metric=5)

    if DistributedManager.distributed:
        grp = DistributedManager.grp
        ws = torch.distributed.get_world_size()
        torch.distributed.all_reduce(gpu_h_agg, op=torch.distributed.ReduceOp.SUM, group=grp)
        torch.distributed.all_reduce(gpu_h_fw, op=torch.distributed.ReduceOp.SUM, group=grp)
        gpu_h_fw /= ws

    print('Time for gradients aggregations: {} [GPU Hours]'.format(gpu_h_agg.item()))
    print('Time for BCSFW: {} [CPU Hours]'.format(gpu_h_fw.item()))

    print('Extract child model')
    child_model, string_model = transform_model_to_mobilenet(model)
    if args.num_gpu > 1:
        child_model = torch.nn.DataParallel(child_model, device_ids=list(range(args.num_gpu)))

    child_model.cuda()
    validate(child_model, loader_eval, validate_loss_fn, args, log_suffix=' child model')

    if saver is not None:
        step = 2 * args.bcfw_steps + 2 if args.fine_tune_alpha else args.bcfw_steps + 1
        saver.save_checkpoint(child_model, optim, args, epoch=step, metric=2)

    model.eval()
    child_model.eval()

    print(f"Computing latency for {string_model}")
    unwrapped_model = model if hasattr(model, 'extract_expected_latency') else model.module
    latency_predicted = unwrapped_model.extract_expected_latency(file_name=args.lut_filename,
                                                                 batch_size=args.lut_measure_batch_size,
                                                                 repeat_measure=args.repeat_measure,
                                                                 target_device=args.target_device)
    latency_measured = measure_time(child_model)
    diff = latency_measured - latency_predicted
    print(f"Latency_predicted={latency_predicted}, latency_measured={latency_measured}, diff={diff}")


def calculate_temperature(t, T0, Tf, tf, policy):
    if policy is None:
        return 1

    if t >= tf:
        return Tf

    T = 1
    if policy == 'linear':
        T = T0 + (Tf - T0) * t / tf

    elif policy == 'exponential':
        r = np.log(Tf / T0)
        T = T0 * np.exp(r * t / tf)

    elif policy == 'cosine':
        T = Tf + 0.5 * (T0 - Tf) * (1 + np.cos(np.pi * t / tf))

    T = max(T, Tf)
    return T


def print_solution(list_alphas, optim, args):
    alpha_attention_vec, _, alpha_grad_vec, alpha_blocks, beta_attention_vec, beta_grad_vec, beta_blocks = \
        flatten_attention_latency_grad_alpha_beta_blocks(list_alphas)

    if args.local_rank != 0:
        return

    print('Solution')
    if args.aggregate_grads_steps is None:
        print('alpha_attention_grads')
        print(np.reshape(alpha_grad_vec, (len(alpha_blocks), -1)))

    print('alpha_attention_vec')
    reshaped = np.reshape(alpha_attention_vec, (len(alpha_blocks), -1)).copy()
    print(reshaped)
    reshaped[reshaped == 0] = 1
    log_p = np.log(reshaped)
    entropies = -np.sum(reshaped * log_p, axis=1) / np.log(reshaped.shape[1])
    entropy_alpha = np.mean(entropies)
    print('alpha attention normalize entropies (mean: {})'.format(entropy_alpha))
    print(entropies)

    print('argmax alpha_attention_vec')
    alpha_argmax_attention = argmax_attention(alpha_attention_vec, alpha_blocks)
    print(np.reshape(alpha_argmax_attention, (len(alpha_blocks), -1)))

    if args.aggregate_grads_steps is None:
        print('beta_attention_grads')
        print(np.reshape(beta_grad_vec, (len(beta_blocks), -1)))

    print('beta_attention_vec')
    reshaped = np.reshape(beta_attention_vec, (len(beta_blocks), -1)).copy()
    print(reshaped)
    reshaped[reshaped == 0] = 1
    log_p = np.log(reshaped)
    entropies = -np.sum(reshaped * log_p, axis=1) / np.log(reshaped.shape[1])
    entropy_beta = np.mean(entropies)
    print('beta attention normalize entropies (mean: {})'.format(entropy_beta))
    print(entropies)

    print('Total entropy: {}'.format(np.mean([entropy_alpha, entropy_beta])))

    print('argmax alpha_attention_vec')
    beta_argmax_attention = argmax_attention(beta_attention_vec, beta_blocks)
    print(np.reshape(beta_argmax_attention, (len(beta_blocks), -1)))

    if 'frank_wolfe' not in args.nas_optimizer:
        return

    check_rounding_constraint(optim, alpha_attention_vec, beta_attention_vec, alpha_blocks, beta_blocks)


def argmax_attention(attention, blocks):
    offset = 0
    argmax_attention_vec = np.zeros_like(attention)
    for block in blocks:
        argmax = np.argmax(attention[offset: offset + block])
        argmax_attention_vec[offset: offset + block] = 0
        argmax_attention_vec[offset + argmax] = 1
        offset += block

    return argmax_attention_vec


def check_rounding_constraint(optim, alpha, beta, alpha_blocks, beta_blocks):
    latency = optim.latency_formula(alpha, beta, optim.fixed_latency)
    print('constraint: {} <= {}'.format(latency, optim.T))
    if latency > optim.T:
        raise Exception('The required latency constraint is infeasible')


    alpha = argmax_attention(alpha, alpha_blocks)
    beta = argmax_attention(beta, beta_blocks)
    latency = optim.latency_formula(alpha, beta, optim.fixed_latency)
    print('argmax constraint: {} <= {}'.format(latency, optim.T))


def compute_and_update_list_alphas(list_alphas, local_rank=0, **kwargs):
    records = compute_aggregate_grads(list_alphas=list_alphas, local_rank=local_rank, **kwargs)
    reduce_and_insert_records(records, list_alphas)
    return records


def reduce_and_insert_records(records, list_alphas):
    if DistributedManager.distributed:
        grp = DistributedManager.grp
        ws = torch.distributed.get_world_size()
        for aggregated_grad in records.values():
            torch.distributed.all_reduce(aggregated_grad, op=torch.distributed.ReduceOp.SUM, group=grp)
            aggregated_grad /= ws

    insert_grads(list_alphas, records)


def extract_grads(list_alphas, agg_steps=0, zero_grad=False, verbose=False):
    if agg_steps == 1:
        return {}

    records = {}
    for e, entry in enumerate(list_alphas):
        key = 'alpha' if 'alpha' in entry else 'beta'
        name = entry['submodules'][0]
        name = get_stage_block_from_name(name, splitted=False) if key is 'alpha' else get_stage_block_from_name(name)[0]
        name = STAGE_BLOCK_DELIMITER.join(['grads', name])
        if key == 'alpha':
            entry['module']._attention_grad = None
            grads = entry['module'].attention_grad
        else:
            grads = entry['module'].beta_attention.grad

        if grads is None or zero_grad:
            grads = torch.zeros_like(entry[key])
            if verbose:
                print('Inserting zero grad for {}'.format(name))
        else:
            grads = grads.detach().clone()
            grads = torch.sum(grads, dim=1) if len(entry[key].shape) < len(grads.shape) else grads

        if DistributedManager.distributed:
            grp = DistributedManager.grp
            torch.distributed.all_reduce(grads, op=torch.distributed.ReduceOp.SUM, group=grp)

        if torch.any(torch.isinf(grads)) or torch.any(torch.isnan(grads)):
            return None

        records[name] = grads

    return records


def insert_grads(list_alphas, records):
    if len(records) == 0:
        return

    for e, entry in enumerate(list_alphas):
        key = 'alpha' if 'alpha' in entry else 'beta'
        name = entry['submodules'][0]
        name = get_stage_block_from_name(name, splitted=False) if key is 'alpha' else get_stage_block_from_name(name)[0]
        name = STAGE_BLOCK_DELIMITER.join(['grads', name])
        recorded_grad = records[name].squeeze()
        recorded_grad = torch.from_numpy(recorded_grad) \
            if not isinstance(recorded_grad, torch.Tensor) else recorded_grad
        if key == 'alpha':
            recorded_grad = recorded_grad.to(
                dtype=entry['module'].attention.dtype, device=entry['module'].attention.device)
            entry['module'].attention_grad = recorded_grad
        else:
            recorded_grad = recorded_grad.to(
                dtype=entry['module'].beta_attention.dtype, device=entry['module'].beta_attention.device)
            entry['module'].beta_attention_grad = recorded_grad

        records[name] = torch.zeros_like(records[name])


def compute_aggregate_grads(list_alphas, model, loss_fn, loader, optimizer, loss_scaler=None, amp_autocast=suppress,
                            local_rank=0, steps=float('Inf'), prefetcher=False, writer=None, target_time_constraint=0,
                            inference_time_limit=0):
    records = extract_grads(list_alphas, zero_grad=True)
    model.eval()
    loss_tot = 0
    for batch_idx in range(steps):
        (input, target) = next(loader)
        model.zero_grad()

        if not prefetcher:
            input, target = input.cuda(), target.cuda()
        with amp_autocast():
            out = model(input)
            loss = loss_fn(out, target)
            loss_tot += loss.item()
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
                if isinstance(optimizer, torch.optim.SGD):
                    loss_scaler(
                        loss, optimizer, parameters=model.parameters(), unscale=False, step=False)
                else:
                    loss_scaler(
                        loss, optimizer, parameters=model.parameters(), unscale=True, add_opt=optim_attention,
                        step=False)
                    loss_scaler.update()
            else:
                loss.backward()
            records_ = extract_grads(list_alphas)

        if records_ is None:
            if batch_idx % 10 == 0 and local_rank == 0:
                logging.info(f"Skiping batch {batch_idx}/{steps}")

            continue

        for name in records.keys():
            records[name] += records_[name]

    loss_tot /= steps
    if writer is not None:
        epoch = optimizer._epoch + 1 if hasattr(optimizer, '_epoch') else None
        writer.add_scalars('loss', {'loss': loss_tot}, global_step=epoch)

    return records


def validate(model, loader, loss_fn, args, log_suffix='', num_iter=-1):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            if num_iter > 0 and batch_idx == num_iter:
                break
            last_batch = batch_idx == last_idx

            if not args.prefetcher:
                input = input.cuda()
                target = target.cuda()

            output = model(input)
            if isinstance(output, (tuple, list)):
                output = output[0]

            # Augmentation reduction
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


def set_argmax_alpha(list_alphas):
    for entry in list_alphas:
        if 'alpha' not in entry:
            continue
        alpha = entry['module'].alpha
        argmax = torch.argmax(alpha)
        alpha.data -= float('inf')
        alpha.data[argmax] = 0


def set_argmax_beta(list_alphas):
    for entry in list_alphas:
        if 'beta' not in entry:
            continue
        beta = entry['module'].beta
        argmax = torch.argmax(beta)
        beta.data -= float('inf')
        beta.data[argmax] = 0


def set_argmax_alpha_beta(list_alphas):
    set_argmax_alpha(list_alphas)
    set_argmax_beta(list_alphas)


if __name__ == '__main__':
    main()
