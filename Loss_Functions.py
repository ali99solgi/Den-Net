import gc
import numpy as np
import os
import shutil
import cv2
import matplotlib.pyplot as plt
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from torch import optim , Tensor
from tqdm import tqdm
from os import listdir
from os.path import isfile, join
from pathlib import Path
from copy import deepcopy
from PIL import Image


def get_num_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = int(sum([np.prod(p.size()) for p in model_parameters]))
    return num_params

def enable_grad(model):
    for param in model.parameters():
        param.requires_grad = True

def disable_grad(model):
    for param in model.parameters():
        param.requires_grad = False

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class DiceLoss(nn.Module):
    def __init__(self, reduction='mean', smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.reduction = reduction
        self.smooth = smooth

    def forward(self, pred, target):

        pred = pred.view(-1).float()
        target = target.view(-1).float()

        inter = 2 * (pred * target).sum()
        sets_sum = pred.sum() + target.sum()

        dice = (inter + self.smooth) / (sets_sum + self.smooth)
        return 1-dice

        ''' if self.reduction == 'mean':
            return (1-dice).mean()
        else:
            return (1-dice).sum() '''
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma, reduction='mean', smooth=1e-6):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.smooth = smooth

    def forward(self, pred, target):

        pred = pred.view(-1).float()
        target = target.view(-1).float()

        BCE = F.binary_cross_entropy(pred, target, reduction=self.reduction)
        BCE_EXP = torch.exp(-BCE)
        focal = self.alpha * (1-BCE_EXP)**self.gamma * BCE
        return  focal

        ''' if self.reduction == 'mean':
            return focal.mean()
        else:
            return focal.sum() '''

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class WBCELoss(nn.Module):
    def __init__(self, alpha=0.5, reduction='mean'):
        super(WBCELoss, self).__init__()
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, pred, target):

        pred = pred.view(-1).float()
        target = target.view(-1).float()
        wbce = -(self.alpha*target*torch.log(pred) + (1-target)*torch.log(1-pred))

        if self.reduction == 'mean':
            return wbce.mean()
        else:
            return wbce.sum()

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class BCELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(BCELoss, self).__init__()
        self.reduction = reduction

    def forward(self, pred, target):

        pred = pred.view(-1).float()
        target = target.view(-1).float()
        bce_loss = nn.BCELoss(reduction=self.reduction)
        bce = bce_loss(pred, target)

        return bce

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=1, reduction='mean', smooth=1e-6):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = 1-alpha
        self.gamma = gamma
        self.reduction = reduction
        self.smooth = smooth

    def forward(self, pred, target):

        pred = pred.view(-1).float()
        target = target.view(-1).float()

        TP = (pred * target).sum()
        FP = ((1-target) * pred).sum()
        FN = (target * (1-pred)).sum()

        tversky = (TP + self.smooth) / (TP + self.alpha*FP + self.beta*FN + self.smooth)
        focal_tversky = (1 - tversky)**self.gamma
        return focal_tversky

        ''' if self.reduction == 'mean':
            return focal_tversky.mean()
        else:
            return focal_tversky.sum() '''

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, pred, target, smooth=1):

        #flatten label and prediction tensors
        pred = pred.view(-1)
        target = target.view(-1)

        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions
        intersection = (pred * target).sum()
        total = (pred + target).sum()
        union = total - intersection

        IoU = (intersection + smooth)/(union + smooth)

        return 1 - IoU

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class criterion(nn.Module):
    def __init__(self,alpha_Focal,gamma_Focal,alpha_FocalTversky,gamma_FocalTversky,alpha_WBCE,reduction,
                 use_loss1,use_loss2,use_loss3,use_loss4,use_loss5,use_loss6):
        super(criterion, self).__init__()

        self.use_loss1 = use_loss1
        self.use_loss2 = use_loss2
        self.use_loss3 = use_loss3
        self.use_loss4 = use_loss4
        self.use_loss5 = use_loss5
        self.use_loss6 = use_loss6

        self.dice_loss = DiceLoss(reduction=reduction)
        self.focal_loss = FocalLoss(alpha=alpha_Focal, gamma=gamma_Focal, reduction=reduction)
        self.WBCE_loss = WBCELoss(alpha=alpha_WBCE, reduction=reduction)
        self.BCE_loss = BCELoss(reduction=reduction)
        self.focal_tversky_loss = FocalTverskyLoss(alpha=alpha_FocalTversky, gamma=gamma_FocalTversky, reduction=reduction)
        self.iou_loss = IoULoss()

    def forward(self, pred, target):

        loss = 0

        if self.use_loss1:
          loss += self.dice_loss(pred, target)

        if self.use_loss2:
          loss += self.focal_loss(pred, target)

        if self.use_loss3:
          loss += self.WBCE_loss(pred, target)

        if self.use_loss4:
          loss += self.BCE_loss(pred, target)

        if self.use_loss5:
          loss += self.focal_tversky_loss(pred, target)

        if self.use_loss6:
          loss += self.iou_loss(pred, target)

        return loss