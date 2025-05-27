# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))


        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def Precision(confusionMatrix):  
    #  返回所有类别的精确率precision  
    precision = np.diag(confusionMatrix) / (confusionMatrix.sum(axis = 0)+1e-8)
    return precision  

def Recall(confusionMatrix):
    #  返回所有类别的召回率recall
    recall = np.diag(confusionMatrix) / (confusionMatrix.sum(axis = 1)+1e-8)
    return recall
  
def F1Score(confusionMatrix):
    precision = np.diag(confusionMatrix) / (confusionMatrix.sum(axis = 0)+1e-8)
    recall = np.diag(confusionMatrix) / (confusionMatrix.sum(axis = 1)+1e-8)
    f1score = 2 * precision * recall / (precision + recall + 1e-8)
    return f1score
