"""
Created by Wang Han on 2020/9/8 22:02.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2020 Wang Han. SCU. All Rights Reserved.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CLS3DLoss(nn.Module):
    def __init__(self, weight):
        super(CLS3DLoss, self).__init__()
        self.weight = weight
        # self.mse = nn.MSELoss(reduction='none')

    def forward(self, input, target):
        # get probility
        prob = torch.sigmoid(input)
        # weight
        front = self.weight * torch.ones_like(target)
        end = torch.ones_like(target)
        weight = torch.where(target != 0, front, end)
        loss = F.binary_cross_entropy(prob, target, weight)
        # loss = torch.mul(weight, self.mse(prob, target)).mean()
        return loss
