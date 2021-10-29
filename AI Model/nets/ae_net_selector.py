"""
Created by Wang Han on 2020/7/21 19:57.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2020 Wang Han. SCU. All Rights Reserved.
"""

import numpy as np

from nets.ae.auto_encoder import AutoEncoder


class AENetSelector(object):

    def __init__(self, config):
        self.config = config

    def get_net(self):
        # loading network parameters
        net_name = self.config['network']['net_name']
        latent_size = self.config['network']['latent_size']
        amcm = self.config['network']['amcm']
        # loading data parameters
        color_channels = self.config['data']['color_channels']
        # loading image size
        image_size = np.array(self.config['train']['aug_trans']['resize']['size'])

        if net_name == 'ae':
            net = AutoEncoder(color_channels=color_channels, image_size=image_size, latent_size=latent_size, amcm=amcm)
        else:
            raise Exception('Not support net: {}.'.format(net_name))

        return net
