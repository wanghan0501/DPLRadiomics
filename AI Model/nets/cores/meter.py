"""
Created by Wang Han on 2018/12/26 22:18.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2018 Wang Han. SCU. All Rights Reserved.
"""

import numpy as np
import torch

__all__ = ['AverageMeter', 'ConfusionMeter']


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ArrayAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, ele_num):
        self.num = ele_num
        self.reset()

    def reset(self):
        self.val = np.zeros([self.num])
        self.avg = np.zeros([self.num])
        self.sum = np.zeros([self.num])
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ConfusionMeter(object):
    """
    The `confusionmeter.ConfusionMeter` constructs a confusion matrix for a multi-class
    classification problems.
    At initialization time, the `k` parameter that indicates the number of classes in the
    classification problem under consideration must be specified.
    The `add(output, target)` method takes as input an NxK (or NxKxHxW) tensor `output`
    that contains the output scores obtained from the model for N examples and K classes
    (H, W is height and width), and a corresponding N-tensor (or NxHxW-tensor) `target`
    that provides the targets for the N examples (H, W is height and width). The targets
    are assumed to be integer values between 0 and K-1.
    The `value(normalized = False)` method returns the confusion matrix in a KxK tensor.
    In the confusion matrix, rows correspond to ground-truth targets and columns correspond
    to predicted targets. Parameter `normalized` (default = `false`) may be specified that
    determines whether or not the confusion matrix is normalized or not when calling value()
    """

    def __init__(self, k):
        super(ConfusionMeter, self).__init__()
        self.classes = k
        self.conf = torch.zeros(k, k)

    def reset(self):
        self.conf.fill_(0)

    def add(self, output, target):
        pr = output.argmax(dim=1).int()
        # pr = (torch.sigmoid(output) > 0.5).int()
        gt = target.int()

        for gt_i in range(self.classes):
            for pr_i in range(self.classes):
                num = (gt == gt_i) * (pr == pr_i)
                self.conf[gt_i][pr_i] += num.sum().cpu()

    def value(self, normalized=False):
        if normalized:
            ret = self.conf / self.conf.sum()
        else:
            ret = self.conf
        return ret.data.cpu().numpy()
