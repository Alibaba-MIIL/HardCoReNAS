import argparse
import random

from timm.models import create_model
from timm.models.mobilenasnet import transform_model_to_mobilenet, measure_time_onnx, measure_time_openvino

parser = argparse.ArgumentParser(description='LUT Generation')
parser.add_argument('--target', '-t', metavar='TARGET', default='onnx',
                    help='Target device to measure latency on (default: onnx)')

model = create_model(
    'mobilenasnet',
    num_classes=1000,
    in_chans=3,
    scriptable=False,
    reduced_exp_ratio=True,
    use_dedicated_pwl_se=False,
    force_sync_gpu=False,
    no_swish=True,
)


def random_one_hot(model):
    for m in model.modules():
        if hasattr(m, 'alpha'):
            m.alpha.data = 0 * m.alpha.data
            index = random.choice(range(len(m.alpha)))
            m.alpha.data[index] = 1
        if hasattr(m, 'beta'):
            for i, b in enumerate(m.beta):
                m.beta[i].data = 0 * m.beta[i].data
                index = random.choice(range(len(m.beta[i])))
                m.beta[i].data[index] = 1


latency_table = []
num_iter = 100

for i in range(num_iter):
    expected_latency = 10000
    # while expected_latency*1e3 > 12:
    random_one_hot(model)
    expected_latency = model.extract_expected_latency(target='onnx', batch_size=1,
                                                      file_name='list_time_bottlenecks_all_open_vino.pkl',
                                                      iterations=100)
    # print(expected_latency)

    model2, string_model = transform_model_to_mobilenet(model)
    model2.eval()
    t = measure_time_openvino(model2)
    print("Done: {}/{}. Expected latency: {:0.2f}[ms] | Measured latency: {:0.2f}[ms]"
          .format(i, num_iter, expected_latency * 1e3, t * 1e3))
    latency_table.append([expected_latency * 1e3, t * 1e3])
    del model2

print(latency_table)
