"""
Created by Wang Han on 2020/6/27 22:14.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2020 Wang Han. SCU. All Rights Reserved.
"""
import numbers

import numpy as np
import torch
from scipy.ndimage import zoom


class Normalize:
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        img = np.array(img).astype(np.float32)
        img = (img - self.mean) / self.std
        return img


class Resize:
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        try:
            scale = np.array(self.size) / np.array(img.shape)
        except:
            import pdb;pdb.set_trace()
        img = zoom(img, scale, order=1)
        return img


class ToTensor:
    def __call__(self, img):
        img = np.expand_dims(img.astype(np.float32), 0)
        img = torch.from_numpy(img).float()
        return img
