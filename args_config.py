import argparse

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--lr', '--learning_rate', default=0.02, type=float,
                        help='initial learning rate')
    parser.add_argument('--noise_mode',  default='sym',
                        help='noise mode: sym or asym')
    parser.add_argument('--alpha', default=4, type=float,
                        help='parameter for Beta')
    parser.add_argument('--lambda_u', default=25, type=float,
                        help='weight for unsupervised loss')
    parser.add_argument('--p_threshold', default=0.5, type=float,
                        help='clean probability threshold')
    parser.add_argument('--T', default=0.5, type=float,
                        help='sharpening temperature')
    parser.add_argument('--num_epochs', default=300, type=int,
                        help='number of training epochs')
    parser.add_argument('--r', default=0.5, type=float,
                        help='noise ratio')
    parser.add_argument('--seed', default=123, type=int,
                        help='random seed')
    parser.add_argument('--gpu_id', default=0, type=int,
                        help='GPU device ID')
    parser.add_argument('--num_class', default=10, type=int,
                        help='number of classes')
    parser.add_argument('--data_path', 
                        default='../../dataset/data/cifar-10-batches-py', 
                        type=str, help='path to dataset')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        help='dataset name')
    parser.add_argument('--run', default=0, type=int,
                        help='run number for experiment tracking')
    parser.add_argument('--warmup_epochs', default=-10, type=int, 
                        help='-10 means use default value')
    parser.add_argument('--num_workers', default=12, type=int,
                        help='number of data loading workers')

    parser.add_argument('--save_prob_log', action='store_true',
                        help='Enable saving probability log data (default: False)')

    # net 2 is vit
    parser.add_argument('--no-net2_vit', action='store_false', dest='net2_vit',
                        help='Disable net2_vit (default: True)')
    parser.add_argument('--clip_model_name', default='ViT-B/32', type=str,
                        help='CLIP model architecture')
    parser.add_argument('--clip_float', action='store_true',
                        help='use float precision for CLIP')
    parser.add_argument('--resume_epoch', default=50, type=int, 
                        help='epoch to resume CLIP visual training')
    parser.add_argument('--vit_learning_rate', default=0.002, 
                        type=float, help='initial learning rate')

    # Unlearning
    parser.add_argument('--no-unlearning', action='store_false', dest='unlearning',
                        help='Disable unlearning (default: True)')
    # Unlearning timing
    parser.add_argument('--unlearning_period', default=10, type=int,
                        help='interval between unlearning phases')
    parser.add_argument('--unlearning_duration', default=3, type=int,
                        help='duration of each unlearning phase')
    parser.add_argument('--unlearning_start', default=50, type=int, 
                        help='epoch to start unlearning')
    # Unlearning SS
    parser.add_argument('--no-current_condition', action='store_false', dest='current_condition',
                        help='Disable current_condition (default: True)')
    parser.add_argument('--no-diff_condition', action='store_false', dest='diff_condition',
                        help='Disable diff_condition (default: True)')
    parser.add_argument('--no-label_condition', action='store_false', dest='label_condition',
                        help='Disable label_condition (default: True)')
    parser.add_argument('--threshold_current', default=0.05, type=float,
                        help='threshold for current condition')
    parser.add_argument('--threshold_diff', default=0.2, type=float,
                        help='threshold for difference condition')
    # Unlearning training
    parser.add_argument('--unlearning_T', default=0.05, type=float,
                        help='temperature for unlearning')
    parser.add_argument('--unlearning_batch_size', default=8, type=int,
                        help='batch size for unlearning')

    # ACD
    parser.add_argument('--no-only_clean', action='store_false', dest='only_clean',
                        help='Disable only_clean (default: True)')
    parser.add_argument('--no-ws_train', action='store_false', dest='ws_train',
                        help='Disable ws_train (default: True)')

    # Use target for network warmup
    parser.add_argument('--wc_target', action='store_true',
                        help='Use target labels for warmup net V')
    parser.add_argument('--wc_alpha', default=0.5, type=float,
                        help='alpha parameter for warmup net V')
    
    # CIFAR-N
    parser.add_argument('--noise_type', default='normal', type=str,
                        help='type of noise for CIFAR-N dataset')

    # Set p_threshold=0.6 when noise ratio r=0.9
    args = parser.parse_args()
    if args.dataset == 'cifar100':
        args.data_path = '../../dataset/data/cifar-100-python/'
        args.num_class = 100
    if args.r == 0.9:  
        args.p_threshold = 0.6

    return args