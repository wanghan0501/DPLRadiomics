"""
Created by Wang Han on 2020/11/8 17:47.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2020 Wang Han. SCU. All Rights Reserved.
"""
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from torchvision.models import vgg16


class PerceptionLoss(_Loss):
    def __init__(self, vgg_layers=16):
        super(PerceptionLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        self.loss_network = nn.Sequential(
            *list(vgg.features)[:vgg_layers]).eval()
        for param in self.loss_network.parameters():
            param.requires_grad = False

    def forward(self, data, reconstructions):
        return torch.mean(torch.pow(self.loss_network(data) - self.loss_network(reconstructions), 2))


class AELoss(_Loss):
    def __init__(self):
        super(AELoss, self).__init__()
        self.mse = nn.MSELoss()
        self.perception = PerceptionLoss()

    def forward(self, data, reconstructions):
        b, c, d, w, h = data.size()
        loss_perception = self.perception(data.view(-1, 3, w, h), reconstructions.view(-1, 3, w, h))
        loss_mse = self.mse(data, reconstructions)
        loss = loss_perception + 10 * loss_mse
        return loss
