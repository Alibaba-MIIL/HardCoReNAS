#!/usr/bin/env python

import argparse

from timm.models.mobilenasnet import *
from timm.models import create_model

parser = argparse.ArgumentParser(description='LUT Generation')
parser.add_argument('--target_device', '-t', metavar='TARGET', default='onnx',
                    help='Target device to measure latency on (default: onnx)')
parser.add_argument('--lut_filename', '-f', metavar='FILENAME', default='lut.pkl',
                    help='The filename of the LUT (default: lut.pkl)')
parser.add_argument('--repeat_measure', type=int , default=100,
                    help='Number of measurements repetitions (default: 100)')
parser.add_argument('--lut_measure_batch_size', type=int, default=1,
                    help='Input batch size for latency LUT measurements (default: 1)')

args = parser.parse_args()

model = create_model(
    'mobilenasnet',
    num_classes=1000,
    in_chans=3,
    scriptable=False,
    reduced_exp_ratio=True,
    use_dedicated_pwl_se=False,
    force_sync_gpu=False,
)


expected_latency = model.extract_expected_latency(target=args.target_device,
                                                  batch_size=args.lut_measure_batch_size,
                                                  file_name=args.lut_filename,
                                                  iterations=args.repeat_measure)

