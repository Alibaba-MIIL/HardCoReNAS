import argparse

def add_nas_to_parser(parser):
    parser.add_argument('--train_elastic_model', action='store_true', default=False,
                        help='Convert model to elastic gumbel model')
    parser.add_argument('--search_elastic_model', type=str2bool, nargs='?', const=True, default=True,
                        help='Search elastic NAS model')
    parser.add_argument('--train_nas', action='store_true', default=False,
                        help='Train the network using nas')
    parser.add_argument('--fixed_alpha', action='store_true', default=False,
                        help='Fixes the alphas')
    parser.add_argument('--force_gs', action='store_true', default=False,
                        help='force gs in evaluation')
    parser.add_argument('--infer_time_constraint', type=float, default=0.0,
                        help='infer_time_constraint')
    parser.add_argument('--target_time_constraint', type=float, default=0.0,
                        help='infer_time_constraint')
    parser.add_argument('--lr_alphas', type=float, default=1e-4, metavar='LR_A',
                        help='learning rate for alphas (default: 1e-4)')
    parser.add_argument('--catchup_epochs', default=0, type=int,
                        help='The number of epochs during which do not update alphas')
    parser.add_argument('--reward_bound', default=float('Inf'), type=float, help='The reward bound')
    parser.add_argument('--sparsity', type=float, default=0,
                        help='Adding a sparsity penalty to the NAS attention with l1 loss')
    parser.add_argument('--zero-one-loss', type=float, default=0,
                        help='Adding a sparsity penalty to the NAS attention using soft penalty')
    parser.add_argument('--parabolic-loss', type=float, default=0,
                        help='Adding a sparsity penalty to the NAS attention with a parabolic loss')
    parser.add_argument('--nas_optimizer', type=str, default='block_frank_wolfe',
                        help='The optimizer used for updating the architecture parameters {block_frank_wolfe, sgd, adam}')
    parser.add_argument('--only_max', action='store_true', default=False,
                        help='takes only the maximum alpha ans set the others to 0')
    parser.add_argument('--force_sm', action='store_true', default=False,
                        help='Force using softmax instead of GS')
    parser.add_argument('--inference_time_limit', type=float, default=0.00125,
                        help='maximum inference time')
    parser.add_argument('--recompute_time_cpu', action='store_true', default=False,
                        help='recompute the inference time on CPU')
    parser.add_argument('--time_measurement_iter', type=int, default=50, metavar='time_measurement_iter',
                        help='Number of iterations to measure the time')
    parser.add_argument('--extended_exp_ratio', dest='extended_exp_ratio', action='store_true',
                        help='use extended exp ratio for search for mobilenet')
    parser.add_argument('--bilevel', action='store_true', default=False,
                        help='Apply bi-level optimization')
    parser.add_argument('--shared_weights', dest='shared_weights', action='store_true',
                        help='use weight sharing  for search for mobilenet')
    parser.add_argument('--hard_backprop_gs', dest='hard_backprop_gs', action='store_true',
                        help='use hard backprop for GS')
    parser.add_argument('--heaviest_network', dest='heaviest_network', action='store_true',
                        help='train the highest network')
    parser.add_argument('--ikd_dividor', type=float, default=10, help='dividor for loss inside the IKD')
    parser.add_argument('--real_KD', dest='real_KD', action='store_true', help='use real KD in the IKD term')
    parser.add_argument('--use_kernel_3', dest='use_kernel_3', action='store_true',
                        help='use kernel of size 3 in mobilenasnet')
    parser.add_argument('--exp_r', type=float, default=6,
                        help='expension ratio')
    parser.add_argument('--depth', type=int, default=4,
                        help='depth of mobilenastnet')
    parser.add_argument('--reduced_exp_ratio', type=str2bool, nargs='?', const=True, default=True,
                        help='use reduced exp ratio')
    parser.add_argument('--hard_backprop', dest='hard_backprop', action='store_true',
                        help='use hard backprop')
    parser.add_argument('--no_privatized_bn', dest='no_privatized_bn', action='store_true',
                        help='Do not use privatized bn')
    parser.add_argument('--use_dedicated_pwl_se', dest='use_dedicated_pwl_se', action='store_true',
                        help='use different pwl convolutions for the output without se')
    parser.add_argument('--force_sync_gpu', dest='force_sync_gpu', action='store_true',
                        help='force_sync_gpu')
    parser.add_argument('--mode_training', default='full1',
                        choices=['static', 'width', 'kernel1', 'kernel2', 'kernel3', 'se', 'depth0', 'depth1',
                                 'depth0_no_kernel', 'depth1_no_kernel', 'kernel1_depth', 'kernel2_depth', 'full0',
                                 'full1'])
    parser.add_argument('--prefer_higher_width_fact', dest='prefer_higher_width_fact', type=float, default=1.0,
                        help='prefers larger width exp ratio in width training')
    parser.add_argument('--prefer_higher_k_fact', dest='prefer_higher_k_fact', type=float, default=1.0,
                        help='prefers larger width exp ratio in width training')
    parser.add_argument('--prefer_higher_beta_fact', dest='prefer_higher_beta_fact', type=float, default=1.0,
                        help='prefers larger depth in training')
    parser.add_argument('--force_se', dest='force_se', action='store_true',
                        help='force to use se in the mobilenetmodel')
    parser.add_argument('--multipath_sampling', type=str2bool, nargs='?', const=True, default=True,
                        help='Multipath sampling')
    parser.add_argument('--freeze_w', dest='freeze_w', action='store_true',
                        help='Freeze the weights and update alpha-beta only')
    parser.add_argument('--w_alpha_update_ratio', dest='w_alpha_update_ratio', type=int, default=1,
                        help='The update frequency ratio between the weights and alpha-beta, '
                             'e.g. 10 means 10 w steps per single alpha-beta step')
    parser.add_argument('--bcfw_steps', dest='bcfw_steps', type=int, default=1000,
                        help='The number of steps for Block Coordinate Frank-Wolfe with frozen w')
    parser.add_argument('--alternate_alpha_beta', action='store_true', default=False,
                        help='Alternate between beta and alpha steps in a round robin manner')
    parser.add_argument('--fixed_grads', action='store_true', default=False,
                        help='The FW expects fixed gradients at all iterations')
    parser.add_argument('--fixed_gamma_step', type=str2bool, nargs='?', const=True, default=True,
                        help='Gamma step always equals to its maximal value if True')
    parser.add_argument('--grads_path', default='./grads_records.pkl', type=str,
                        help='The path of the aggregated gradients pickle file')
    parser.add_argument('--aggregate_grads_steps', dest='aggregate_grads_steps', type=int, default=1,
                        help='The number of batches to aggregate gradients over per FW step')
    parser.add_argument('--init_to_biggest', action='store_true', default=False,
                        help='Initialize the structure to the biggest possible')
    parser.add_argument('--init_to_smallest', action='store_true', default=False,
                        help='Initialize the structure to the smallest possible')
    parser.add_argument('--init_to_biggest_alpha', action='store_true', default=False,
                        help='Initialize the structure to the biggest possible alpha')
    parser.add_argument('--fine_tune_alpha', action='store_true', default=False,
                        help='Fine tune alpha only after beta is fixed to its one-hot')
    parser.add_argument('--qc_init', type=str2bool, nargs='?', const=True, default=True,
                        help='Initialize to a balanced structure through quadratic programming if True')
    parser.add_argument('--max_gamma_step', default=0.002, type=float,
                        help='The maximal gamma step size for the BCFW')
    parser.add_argument('--sfw_momentum', default=0.9, type=float,
                        help='The momentum of S[tochastic]FW')
    parser.add_argument('--convert_model_to_mobilenet', action='store_true', default=False,
                        help='Convert the model to a mobilenet model')
    parser.add_argument('--transform_model_to_mobilenet', action='store_true', default=False,
                        help='force transform_model_to_mobilenet')
    parser.add_argument('--no_weight_loading', action='store_true', default=False,
                        help='do not load weight of parent nas model')
    parser.add_argument('--use_softmax', action='store_true', default=False,
                        help='Use the softmax instead of gumble softmax')
    parser.add_argument('--annealing_policy', default=None, type=str,
                        help='The temperature for the softmax')
    parser.add_argument('--init_temperature', default=1., type=float,
                        help='The initial temperature for the softmax')
    parser.add_argument('--final_temperature', default=1., type=float,
                        help='The final temperature for the softmax')
    parser.add_argument('--temperature_annealing_period', default=1., type=float,
                        help='The portion of the search to reach the final temperature, e.g. 0.5 for 50% of the search')
    parser.add_argument('--detach_gs', action='store_true', default=False,
                        help='Detach the GS from the computational graph')
    parser.add_argument('--start_with_alpha', action='store_true', default=False,
                        help='`start the optimization with alpha update')
    parser.add_argument('--steps_per_grad', default=1, type=int,
                        help='The number of alpha-beta steps per single gradients backprop')
    parser.add_argument('--mobilenasnet_er', choices=[3, 4, 6], type=int)
    parser.add_argument('--mobilenasnet_kernel', choices=[3, 5], type=int)
    parser.add_argument('--mobilenasnet_depth', choices=[2, 3, 4], type=int)
    parser.add_argument('--set_alpha_beta', action='store_true', default=False,
                        help='set the alpha and beta for mobilenasnet')
    parser.add_argument('--target_device', '-t', metavar='TARGET', default='cpu',
                        help='Target device to measure latency on (cpu, gpu or onnx - default: cpu)')
    parser.add_argument('--lut_filename', '-f', metavar='FILENAME', default='lut.pkl',
                        help='The filename of the LUT (default: lut.pkl)')
    parser.add_argument('--repeat_measure', type=int, default=100,
                        help='Number of measurements repetitions (default: 100)')
    parser.add_argument('--lut_measure_batch_size', type=int, default=1,
                        help='Number of measurements repetitions (default: 100)')
    parser.add_argument('--mobilenet_string', type=str, default='')

    parser.add_argument('--no_swish', action='store_true', default=False,
                        help='use ReLU in the entire network')
    parser.add_argument('--latency_corrective_slope', default=1., type=float,
                        help='The slope to correct the latency formula from the linear regression')
    parser.add_argument('--latency_corrective_intercept', default=1., type=float,
                        help='The intercept to correct the latency formlula from the linear regression')
    parser.add_argument('--use_swish', action='store_true', default=False,
                        help='use ReLU in the entire network')
    parser.add_argument('--gamma_knowledge', type=float, default=10,
                        help='factor for distillation penalty')


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')