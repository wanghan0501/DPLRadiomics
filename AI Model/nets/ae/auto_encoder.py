"""
Created by Wang Han on 2020/11/10 16:32.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2020 Wang Han. SCU. All Rights Reserved.
"""
import numpy as np
from torch import nn


class AutoEncoder(nn.Module):
    def __init__(self, color_channels, image_size, latent_size, amcm):
        super(AutoEncoder, self).__init__()
        self.color_channels = color_channels
        self.image_size = image_size
        self.latent_size = latent_size
        self.amcm = amcm
        self.conv_size = (image_size / 16).astype('int')

        self.encoder = nn.Sequential(
            nn.Conv3d(color_channels, amcm, 4, 2, 1, bias=False),
            nn.BatchNorm3d(amcm),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(amcm, amcm * 2, 4, 2, 1, bias=False),
            nn.BatchNorm3d(amcm * 2),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(amcm * 2, amcm * 4, 4, 2, 1, bias=False),
            nn.BatchNorm3d(amcm * 4),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(amcm * 4, amcm * 8, 4, 2, 1, bias=False),
            nn.BatchNorm3d(amcm * 8),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(int((amcm * 8) * np.prod(self.conv_size)), self.latent_size),  # 6*6
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(self.latent_size, int((amcm * 8) * np.prod(self.conv_size))),  # 6*6
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d((amcm * 8), amcm * 4, 4, 2, 1, bias=False),
            nn.BatchNorm3d(amcm * 4),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.ConvTranspose3d(amcm * 4, amcm * 2, 4, 2, 1, bias=False),
            nn.BatchNorm3d(amcm * 2),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.ConvTranspose3d(amcm * 2, amcm, 4, 2, 1, bias=False),
            nn.BatchNorm3d(amcm),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.ConvTranspose3d(amcm, color_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, input, is_z=False):
        bs = input.size(0)
        feature = self.encoder(input)
        z = self.fc1(feature.view(bs, -1))
        feature = self.fc2(z).reshape(bs, self.amcm * 8, self.conv_size[0], self.conv_size[1], self.conv_size[2])
        output = self.decoder(feature).view(bs, self.color_channels, self.image_size[0], self.image_size[1],
                                            self.image_size[2])
        if is_z:
            return output, z
        else:
            return output
