# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import functools

import numpy as np

import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F
import math
from einops import rearrange, repeat

from utils.grl import ReverseLayerF 

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(-1, -2))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        # attn = self.dropout(F.softmax(attn, dim=-1))
        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v)

        return output, attn
    


class StackedGRUWithDropout(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.5):
        super(StackedGRUWithDropout, self).__init__()

        self.gru = nn.GRU(input_size=input_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     batch_first=True,
                                     dropout=dropout,
                                     bidirectional=True,
                                     )
    def forward(self, x):
        # 通过整个堆叠的GRU层
        self.gru.flatten_parameters()
        output, _ = self.gru(x)

        return output
    
class Net(nn.Module):

    def __init__(self,cfg,**kwargs):
        super(Net, self).__init__()
        self.in_chans = cfg["MODEL"]["IN_CHANNEL"]
        self.T = cfg["MODEL"]["T"]
        self.USE_T = cfg["MODEL"]["USE_T"]
        self.num_class = cfg["MODEL"]["NUM_CLASSES"]
        self.classifier = nn.Linear(256*self.USE_T,self.num_class)
        self.gru1 = StackedGRUWithDropout(self.in_chans, hidden_size=128) #2层gru
        self.gru2 = StackedGRUWithDropout(256, hidden_size=128)
        self.attn = ScaledDotProductAttention(temperature=1)

    def forward(self,x,ndvi=None):
        x = x[:,:,:self.USE_T]

        out1= self.gru1(x.permute(0,2,1))
        out2 = self.gru2(out1)
        x,A = self.attn(out2,out2,out2)
        out3 = x+out2

        x_c = self.classifier(out3.flatten(1))
        out3= out3.reshape(out3.shape[0],-1)

        return x_c,out1,out3,ndvi,A



    def init_weights(self, pretrained='',):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            for k, _ in pretrained_dict.items():
                logger.info(
                    '=> loading {} pretrained model {}'.format(k, pretrained))
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)


def get_cls_net(config, **kwargs):
    model = Net(config, **kwargs)
    model.init_weights()
    return model
