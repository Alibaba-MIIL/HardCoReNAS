import torch.utils.model_zoo as model_zoo
from .registry import is_model, is_model_in_modules, model_entrypoint
from .helpers import load_checkpoint
from .layers import set_layer_config


def create_model(
    model_name,
    pretrained=False,
    num_classes=1000,
    in_chans=3,
    checkpoint_path='',
    scriptable=None,
    exportable=None,
    no_jit=None,
    strict=True,
    **kwargs):
    """Create a model

    Args:
        model_name (str): name of model to instantiate
        pretrained (bool): load pretrained ImageNet-1k weights if true
        num_classes (int): number of classes for final fully connected layer (default: 1000)
        in_chans (int): number of input channels / colors (default: 3)
        checkpoint_path (str): path of checkpoint to load after model is initialized
        scriptable (bool): set layer config so that model is jit scriptable (not working for all models yet)
        exportable (bool): set layer config so that model is traceable / ONNX exportable (not fully impl/obeyed yet)
        no_jit (bool): set layer config so that model doesn't utilize jit scripted layers (so far activations only)

    Keyword Args:
        drop_rate (float): dropout rate for training (default: 0.0)
        global_pool (str): global pool type (default: 'avg')
        **: other kwargs are model specific
    """
    model_args = dict(pretrained=pretrained, num_classes=num_classes, in_chans=in_chans)

    # Only EfficientNet and MobileNetV3 models have support for batchnorm params or drop_connect_rate passed as args
    is_efficientnet = is_model_in_modules(model_name, ['efficientnet', 'mobilenetv3'])
    is_resnet = is_model_in_modules(model_name, ['resnet', 'resnext'])
    if not is_efficientnet:
        kwargs.pop('bn_tf', None)
        kwargs.pop('bn_momentum', None)
        kwargs.pop('bn_eps', None)

    if not is_resnet:
        kwargs.pop('no_skip', None)
    if model_name != 'resnetcustom' and model_name != 'ecaresnetdcustom' and model_name != 'ecaresnetcustom':
        if 'resnet_structure' in kwargs:
            kwargs.pop('resnet_structure')
            kwargs.pop('resnet_block')

    if 'mobilenasnet' not in model_name:
        if 'heaviest_network' in kwargs:
            kwargs.pop('heaviest_network')
        if 'use_kernel_3' in kwargs:
            kwargs.pop('use_kernel_3')
        if 'exp_r' in kwargs:
            kwargs.pop('exp_r')
        if 'depth' in kwargs:
            kwargs.pop('depth')
        if 'reduced_exp_ratio' in kwargs:
            kwargs.pop('reduced_exp_ratio')
        if 'use_dedicated_pwl_se' in kwargs:
            kwargs.pop('use_dedicated_pwl_se')
        if 'force_sync_gpu' in kwargs:
            kwargs.pop('force_sync_gpu')
        if 'no_privatized_bn' in kwargs:
            kwargs.pop('no_privatized_bn')
        if 'multipath_sampling' in kwargs:
            kwargs.pop('multipath_sampling')
        if 'use_softmax' in kwargs:
            kwargs.pop('use_softmax')
        if 'detach_gs' in kwargs:
            kwargs.pop('detach_gs')
        if 'mobilenet_string' in kwargs:
            kwargs.pop('mobilenet_string')
        if 'search_mode' in kwargs:
            kwargs.pop('search_mode')
        if 'no_swish' in kwargs:
            kwargs.pop('no_swish')
        if 'use_swish' in kwargs:
            kwargs.pop('use_swish')

    # Parameters that aren't supported by all models should default to None in command line args,
    # remove them if they are present and not set so that non-supporting models don't break.
    if kwargs.get('drop_block_rate', None) is None:
        kwargs.pop('drop_block_rate', None)

    # handle backwards compat with drop_connect -> drop_path change
    drop_connect_rate = kwargs.pop('drop_connect_rate', None)
    if drop_connect_rate is not None and kwargs.get('drop_path_rate', None) is None:
        print("WARNING: 'drop_connect' as an argument is deprecated, please use 'drop_path'."
              " Setting drop_path to %f." % drop_connect_rate)
        kwargs['drop_path_rate'] = drop_connect_rate

    if kwargs.get('drop_path_rate', None) is None:
        kwargs.pop('drop_path_rate', None)

    with set_layer_config(scriptable=scriptable, exportable=exportable, no_jit=no_jit):
        if is_model(model_name):
            create_fn = model_entrypoint(model_name)
            model = create_fn(**model_args, **kwargs)
        else:
            raise RuntimeError('Unknown model (%s)' % model_name)

    if checkpoint_path:
        load_checkpoint(model, checkpoint_path, strict=strict, use_ema=True)

    return model
