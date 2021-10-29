"""
Created by Wang Han on 2019/3/29 14:43.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2019 Wang Han. SCU. All Rights Reserved.
"""
import argparse
import random

import torch

from utils.gpu_util import set_gpu
from utils.parse_util import parse_yaml
from utils.str_util import str2bool

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch DCH_AI')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed for training. default=42')
    parser.add_argument('--use_cuda', default='true', type=str,
                        help='whether use cuda. default: true')
    parser.add_argument('--use_parallel', default='false', type=str,
                        help='whether use parallel. default: false')
    parser.add_argument('--gpu', default='all', type=str,
                        help='use gpu device. default: all')
    parser.add_argument('--model', default='ae', type=str,
                        choices=['ae'],
                        help='which model used. default: seg')
    parser.add_argument('--logdir', default=None, type=str,
                        help='which logdir used. default: None')
    parser.add_argument('--train_sample_csv', default=None, type=str,
                        help='train sample csv file used. default: None')
    parser.add_argument('--eval_sample_csv', default=None, type=str,
                        help='eval sample csv file used. default: None')
    parser.add_argument('--config', default='cfgs/crc_ae.yaml', type=str,
                        help='configuration file. default: cfgs/crc_ae.yaml')

    args = parser.parse_args()

    num_gpus = set_gpu(args.gpu)
    # set seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    config = parse_yaml(args.config)
    network_params = config['network']

    network_params['seed'] = args.seed
    network_params['device'] = "cuda" if str2bool(args.use_cuda) else "cpu"

    network_params['use_parallel'] = str2bool(args.use_parallel)
    network_params['num_gpus'] = num_gpus
    if num_gpus > 1:
        network_params['use_parallel'] = True
    config['network'] = network_params

    train_params = config['train']
    train_params['batch_size'] = train_params['batch_size'] * num_gpus
    train_params['num_workers'] = train_params['num_workers'] * num_gpus
    config['train'] = train_params

    eval_params = config['eval']
    eval_params['batch_size'] = eval_params['batch_size'] * num_gpus
    eval_params['num_workers'] = eval_params['num_workers'] * num_gpus
    config['eval'] = eval_params

    if args.model == 'ae':
        from models.crc.crc_ae_model import Model


    model = Model(config)
    net_name = network_params['net_name']
    if network_params['use_pretrained']:
        model.load_pretrained(network_params['pretrained_path'][net_name])
    model.run()
