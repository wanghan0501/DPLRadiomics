"""
Created by Wang Han on 2020/8/4 10:10.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2020 Wang Han. SCU. All Rights Reserved.
"""
import torch
from torch.nn.modules.loss import _Loss


class SoftDiceLoss(_Loss):
    '''
    Soft_Dice = 2*|A * B| / (|A*A| + |B*B| + eps)
    eps is a small constant to avoid zero division,
    '''

    def __init__(self, *args, **kwargs):
        super(SoftDiceLoss, self).__init__()

    def forward(self, y_pred, y_true, eps=1e-8):
        y_pred = torch.sigmoid(y_pred)
        intersection = torch.sum(torch.mul(y_pred, y_true))
        union = torch.sum(torch.mul(y_pred, y_pred)) + torch.sum(torch.mul(y_true, y_true)) + eps

        dice = 2 * intersection / union
        dice_loss = 1 - dice
        return dice_loss


class DiceLoss(_Loss):
    '''
    Dice = 2*|A * B| / (|A| + |B| + eps)
    eps is a small constant to avoid zero division,
    '''

    def __init__(self, *args, **kwargs):
        super(DiceLoss, self).__init__()

    def forward(self, y_pred, y_true, eps=1e-8):
        intersection = torch.sum(torch.mul(y_pred, y_true))
        union = torch.sum(y_pred) + torch.sum(y_true) + eps

        dice = 2 * intersection / union
        dice_loss = 1 - dice

        return dice_loss
