#!/usr/bin/env python
""" ImageNet Validation Script

This is intended to be a lean and easily modifiable ImageNet validation script for evaluating pretrained
models or training checkpoints against ImageNet or similarly organized image datasets. It prioritizes
canonical PyTorch, standard Python style, and good performance. Repurpose as you see fit.

Hacked together by Ross Wightman (https://github.com/rwightman)
"""
import csv
import glob
import logging
import os
from collections import OrderedDict

import torch.nn.parallel
import torch.utils.model_zoo as model_zoo
from torch import nn

from external.utils_pruning import compute_flops

try:
    from apex import amp

    has_apex = True
except ImportError:
    has_apex = False

from timm.models import create_model, apply_test_time_pool, load_checkpoint, is_model, list_models
from timm.data import Dataset, DatasetTar, create_loader, resolve_data_config
from timm.utils import accuracy, AverageMeter, natural_key, setup_default_logging
from timm.models.helpers import fuse_bn
from external.nas_parser import *
from timm.models.mobilenasnet import transform_model_to_mobilenet, measure_time, measure_time_onnx
import time

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--model', '-m', metavar='MODEL', default='mobilenasnet',
                    help='model architecture (default: mobienasnet)')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--img-size', default=None, type=int,
                    metavar='N', help='Input image dimension, uses model default if empty')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop pct')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--num-classes', type=int, default=1000,
                    help='Number classes in dataset')
parser.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                    help='path to class to idx mapping file (default: "")')
parser.add_argument('--log-freq', default=10, type=int,
                    metavar='N', help='batch logging frequency (default: 10)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--no-test-pool', dest='no_test_pool', action='store_true',
                    help='disable test time pool')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--amp', action='store_true', default=False,
                    help='Use AMP mixed precision')
parser.add_argument('--tf-preprocessing', action='store_true', default=False,
                    help='Use Tensorflow preprocessing pipeline (require CPU TF installed')
parser.add_argument('--use-ema', dest='use_ema', action='store_true',
                    help='use ema version of weights if present')
parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='convert model torchscript for inference')
parser.add_argument('--results-file', default='', type=str, metavar='FILENAME',
                    help='Output csv file for validation results (summary)')

parser.add_argument('--fuse-bn', type=str2bool, nargs='?', const=True, default=True,
                    help='fuse the bn for the model model')
parser.add_argument('--squish', action='store_true', default=False, help='use squish for resize input image')
parser.add_argument('--normalize_weights', action='store_true', default=False, help='Normalize the weights')
parser.add_argument('--resnet_structure', type=int, nargs='+', default=[3, 4, 6, 3], metavar='resnetstruct',
                    help='custom resnet structure')
parser.add_argument('--resnet_block', default='Bottleneck', type=str, metavar='block',
                    help='custom resnet block')



add_nas_to_parser(parser)

def validate(args):
    args.pretrained = args.pretrained or (not args.checkpoint)
    args.prefetcher = not args.no_prefetcher
    if os.path.splitext(args.data)[1] == '.tar' and os.path.isfile(args.data):
        dataset = DatasetTar(args.data, load_bytes=args.tf_preprocessing, class_map=args.class_map)
    else:
        dataset = Dataset(args.data, load_bytes=args.tf_preprocessing, class_map=args.class_map)
    logging.info(f'Validation data has {len(dataset)} images')
    args.num_classes = len(dataset.class_to_idx)
    logging.info(f'setting num classes to {args.num_classes}')

    # create model
    model = create_model(
        args.model,
        num_classes=args.num_classes,
        in_chans=3,
        pretrained=args.pretrained,
        scriptable=args.torchscript,
        resnet_structure=args.resnet_structure,
        resnet_block=args.resnet_block,
        heaviest_network=args.heaviest_network,
        use_kernel_3=args.use_kernel_3,
        exp_r=args.exp_r,
        depth=args.depth,
        reduced_exp_ratio=args.reduced_exp_ratio,
        use_dedicated_pwl_se=args.use_dedicated_pwl_se,
        multipath_sampling=args.multipath_sampling,
        force_sync_gpu=args.force_sync_gpu,
        mobilenet_string=args.mobilenet_string if not args.transform_model_to_mobilenet else '',
        no_swish=args.no_swish,
        use_swish=args.use_swish
    )
    data_config = resolve_data_config(vars(args), model=model)
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, True, strict=True)

    if 'mobilenasnet' in args.model and args.transform_model_to_mobilenet:
        model.eval()
        expected_latency = model.extract_expected_latency(file_name=args.lut_filename,
                                                          batch_size=args.lut_measure_batch_size,
                                                          iterations=args.repeat_measure,
                                                          target=args.target_device)
        model.eval()
        model2, string_model = transform_model_to_mobilenet(model, mobilenet_string=args.mobilenet_string)
        del model
        model = model2
        model.eval()
        print('Model converted. Expected latency: {:0.2f}[ms]'.format(expected_latency * 1e3))

    elif args.normalize_weights:
        IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
        std = torch.tensor(IMAGENET_DEFAULT_STD).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        mean = torch.tensor(IMAGENET_DEFAULT_MEAN).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        W = model.conv_stem.weight.data
        bnw = model.bn1.weight.data
        bnb = model.bn1.bias.data
        model.conv_stem.weight.data = W / std
        bias = - bnw.data * (W.sum(dim=[-1, -2]) @ (mean / std).squeeze()) / (
            torch.sqrt(model.bn1.running_var + model.bn1.eps))
        model.bn1.bias.data = bnb + bias

    if args.fuse_bn:
        model = fuse_bn(model)

    if args.target_device == 'gpu':
        measure_time(model, batch_size=64, target='gpu')
        t = measure_time(model, batch_size=64, target='gpu')

    elif args.target_device == 'onnx':
        t = measure_time_onnx(model)

    else:
        measure_time(model)
        t = measure_time(model)

    param_count = sum([m.numel() for m in model.parameters()])
    flops = compute_flops(model, data_config['input_size'])
    logging.info('Model {} created, param count: {}, flops: {}, Measured latency ({}): {:0.2f}[ms]'
                 .format(args.model, param_count, flops / 1e9, args.target_device, t * 1e3))

    data_config = resolve_data_config(vars(args), model=model, verbose=False)
    model, test_time_pool = apply_test_time_pool(model, data_config, args)

    if args.torchscript:
        torch.jit.optimized_execution(True)
        model = torch.jit.script(model)

    if args.amp:
        model = amp.initialize(model.cuda(), opt_level='O1')

    else:
        model = model.cuda()

    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu)))

    criterion = nn.CrossEntropyLoss().cuda()

    crop_pct = 1.0 if test_time_pool else data_config['crop_pct']
    loader = create_loader(
        dataset,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        crop_pct=crop_pct,
        pin_memory=args.pin_mem,
        tf_preprocessing=args.tf_preprocessing,
        squish=args.squish,
    )

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.cuda()
    model.eval()
    with torch.no_grad():
        # warmup, reduce variability of first batch time, especially for comparing torchscript vs non
        input = torch.randn((args.batch_size,) + data_config['input_size']).cuda()
        model(input)
        end = time.time()
        for i, (input, target) in enumerate(loader):
            if i == 0:
                end = time.time()

            if args.no_prefetcher:
                target = target.cuda()
                input = input.cuda()

            if args.amp:
                input = input.half()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            k = min(5, args.num_classes)
            acc1, acc5 = accuracy(output.data, target, topk=(1, k))

            losses.update(loss.item(), input.size(0))
            top1.update(acc1.item(), input.size(0))
            top5.update(acc5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.log_freq == 0:
                logging.info(
                    'Test: [{0:>4d}/{1}]  '
                    'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.3f} ({top1.avg:>7.3f})  '
                    'Acc@5: {top5.val:>7.3f} ({top5.avg:>7.3f})'.format(
                        i, len(loader), batch_time=batch_time,
                        rate_avg=input.size(0) / batch_time.avg,
                        loss=losses, top1=top1, top5=top5))

    results = OrderedDict(
        top1=round(top1.avg, 4), top1_err=round(100 - top1.avg, 4),
        top5=round(top5.avg, 4), top5_err=round(100 - top5.avg, 4),
        param_count=round(param_count / 1e6, 2),
        img_size=data_config['input_size'][-1],
        cropt_pct=crop_pct,
        interpolation=data_config['interpolation'])

    logging.info(' * Acc@1 {:.3f} ({:.3f}) Acc@5 {:.3f} ({:.3f})'.format(
        results['top1'], results['top1_err'], results['top5'], results['top5_err']))

    return results


def main():
    setup_default_logging()
    args = parser.parse_args()
    model_cfgs = []
    model_names = []
    if os.path.isdir(args.checkpoint):
        # validate all checkpoints in a path with same model
        checkpoints = glob.glob(args.checkpoint + '/*.pth.tar')
        checkpoints += glob.glob(args.checkpoint + '/*.pth')
        model_names = list_models(args.model)
        model_cfgs = [(args.model, c) for c in sorted(checkpoints, key=natural_key)]

    else:
        if args.model == 'all':
            # validate all models in a list of names with pretrained checkpoints
            args.pretrained = True
            model_names = list_models(pretrained=True)
            model_cfgs = [(n, '') for n in model_names]

        elif not is_model(args.model):
            # model name doesn't exist, try as wildcard filter
            model_names = list_models(args.model)
            model_cfgs = [(n, '') for n in model_names]

    if len(model_cfgs):
        results_file = args.results_file or './results-all.csv'
        logging.info('Running bulk validation on these pretrained models: {}'.format(', '.join(model_names)))
        results = []
        try:
            start_batch_size = args.batch_size
            for m, c in model_cfgs:
                batch_size = start_batch_size
                args.model = m
                args.checkpoint = c
                result = OrderedDict(model=args.model)
                r = {}
                while not r and batch_size >= args.num_gpu:
                    try:
                        args.batch_size = batch_size
                        print('Validating with batch size: %d' % args.batch_size)
                        r = validate(args)

                    except RuntimeError as e:
                        if batch_size <= args.num_gpu:
                            print("Validation failed with no ability to reduce batch size. Exiting.")
                            raise e

                        batch_size = max(batch_size // 2, args.num_gpu)
                        print("Validation failed, reducing batch size by 50%")
                        torch.cuda.empty_cache()

                result.update(r)
                if args.checkpoint:
                    result['checkpoint'] = args.checkpoint

                results.append(result)

        except KeyboardInterrupt as e:
            pass

        results = sorted(results, key=lambda x: x['top1'], reverse=True)
        if len(results):
            write_results(results_file, results)

    else:
        validate(args)


def write_results(results_file, results):
    with open(results_file, mode='w') as cf:
        dw = csv.DictWriter(cf, fieldnames=results[0].keys())
        dw.writeheader()
        for r in results:
            dw.writerow(r)

        cf.flush()


if __name__ == '__main__':
    main()
