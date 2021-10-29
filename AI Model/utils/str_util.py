"""
Created by Wang Han on 2020/8/5 11:17.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2020 Wang Han. SCU. All Rights Reserved.
"""
import argparse


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')
