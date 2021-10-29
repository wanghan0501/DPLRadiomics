"""
Created by Wang Han on 2019/3/30 13:30.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2019 Wang Han. SCU. All Rights Reserved.
"""

import torch.nn as nn


class NormSelector(object):

    @staticmethod
    def Norm3d(norm_type=None):
        if norm_type == 'batchnorm':
            return nn.BatchNorm3d

        elif norm_type == 'groupnorm':
            return nn.GroupNorm

        elif norm_type == 'instancenorm':
            return nn.InstanceNorm3d

        elif norm_type == 'switchnorm':
            from nets.cores.switchnorm.switchnorm import SwitchNorm3d
            return SwitchNorm3d

        else:
            raise Exception('Not support BN type: {}.'.format(norm_type))

    @staticmethod
    def Norm2d(norm_type=None):
        if norm_type == 'batchnorm':
            return nn.BatchNorm2d

        elif norm_type == 'groupnorm':
            return nn.GroupNorm

        elif norm_type == 'instancenorm':
            return nn.InstanceNorm3d

        elif norm_type == 'switchnorm':
            from nets.cores.switchnorm.switchnorm import SwitchNorm2d

            return SwitchNorm2d

        else:
            raise Exception('Not support BN type: {}.'.format(norm_type))

    @staticmethod
    def Norm1d(norm_type=None):
        if norm_type == 'batchnorm':
            return nn.BatchNorm1d

        elif norm_type == 'groupnorm':
            return nn.GroupNorm

        elif norm_type == 'instancenorm':
            return nn.InstanceNorm1d

        elif norm_type == 'switchnorm':
            from nets.cores.switchnorm.switchnorm import SwitchNorm1d

            return SwitchNorm1d

        else:
            raise Exception('Not support BN type: {}.'.format(norm_type))
