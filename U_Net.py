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


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, k1=1, p1=0, k2=3, p2=1, device='cuda', dtype=torch.float32, dropout=0.1, first_norm=True):
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels

        if first_norm==False:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=k1, padding=p1, bias=False, device=device, dtype=dtype),
                nn.BatchNorm2d(mid_channels, device=device, dtype=dtype), nn.ReLU(inplace=True), nn.Dropout(dropout),
                nn.Conv2d(mid_channels, out_channels, kernel_size=k2, padding=p2, bias=False, device=device, dtype=dtype),
                nn.BatchNorm2d(out_channels, device=device, dtype=dtype), nn.ReLU(inplace=True), nn.Dropout(dropout))
        else:
            self.double_conv = nn.Sequential(
                nn.BatchNorm2d(in_channels, device=device, dtype=dtype),
                nn.Conv2d(in_channels, mid_channels, kernel_size=k1, padding=p1, bias=False, device=device, dtype=dtype),
                nn.BatchNorm2d(mid_channels, device=device, dtype=dtype), nn.ReLU(inplace=True), nn.Dropout(dropout),
                nn.Conv2d(mid_channels, out_channels, kernel_size=k2, padding=p2, bias=False, device=device, dtype=dtype),
                nn.BatchNorm2d(out_channels, device=device, dtype=dtype), nn.ReLU(inplace=True), nn.Dropout(dropout))

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, k1=1, p1=0, k2=3, p2=1, device='cuda', dtype=torch.float32, dropout=0.1, first_norm=True):
        super().__init__()

        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels, k1=k1, p1=p1, k2=k2, p2=p2, device=device , dtype=dtype, dropout=dropout, first_norm=first_norm))

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, k1=1, p1=0, k2=3, p2=1, device='cuda', dtype=torch.float32, dropout=0.1, first_norm=True):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2, device=device, dtype=dtype)
        self.conv = DoubleConv(in_channels, out_channels, k1=k1, p1=p1, k2=k2, p2=p2, device=device, dtype=dtype, dropout=dropout, first_norm=first_norm)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, device='cuda', dtype=torch.float32):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, device=device, dtype=dtype)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, d1=32, d2=64, d3=128, d4=256, d5=512, kernel1=1, padding1=0, kernel2=3, padding2=1, device='cuda', dtype=torch.float32, dropout=0.1, first_norm=True):
        super(UNet, self).__init__()

        self.in_layer     = DoubleConv(in_channels, d1, k1=kernel1, p1=padding1, k2=kernel2, p2=padding2, device=device, dtype=dtype, dropout=dropout, first_norm=first_norm)
        self.down1        = Down(d1, d2, k1=kernel1, p1=padding1, k2=kernel2, p2=padding2, device=device, dtype=dtype, dropout=dropout, first_norm=first_norm)
        self.down2        = Down(d2, d3, k1=kernel1, p1=padding1, k2=kernel2, p2=padding2, device=device, dtype=dtype, dropout=dropout, first_norm=first_norm)
        self.down3        = Down(d3, d4, k1=kernel1, p1=padding1, k2=kernel2, p2=padding2, device=device, dtype=dtype, dropout=dropout, first_norm=first_norm)
        self.down4        = Down(d4, d5, k1=kernel1, p1=padding1, k2=kernel2, p2=padding2, device=device, dtype=dtype, dropout=dropout, first_norm=first_norm)

        self.up1          = Up  (d5, d4, k1=kernel1, p1=padding1, k2=kernel2, p2=padding2, device=device, dtype=dtype, dropout=dropout, first_norm=first_norm)
        self.up2          = Up  (d4, d3, k1=kernel1, p1=padding1, k2=kernel2, p2=padding2, device=device, dtype=dtype, dropout=dropout, first_norm=first_norm)
        self.up3          = Up  (d3, d2, k1=kernel1, p1=padding1, k2=kernel2, p2=padding2, device=device, dtype=dtype, dropout=dropout, first_norm=first_norm)
        self.up4          = Up  (d2, d1, k1=kernel1, p1=padding1, k2=kernel2, p2=padding2, device=device, dtype=dtype, dropout=dropout, first_norm=first_norm)
        self.out_layer    = OutConv(d1, out_channels, device=device, dtype=dtype)

        self.sigmoid      = nn.Sigmoid()

    def forward(self, x):

        x1  = self.in_layer(x)
        x2  = self.down1(x1)
        x3  = self.down2(x2)
        x4  = self.down3(x3)
        x5  = self.down4(x4)

        x   = self.up1(x5 , x4)
        x   = self.up2(x  , x3)
        x   = self.up3(x  , x2)
        x   = self.up4(x  , x1)
        output = self.out_layer(x)

        output = self.sigmoid(output)

        return output
    
#-------------------------------------------------------------------------------------------------------------------------------------------------------

class SingleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, k1=1, p1=0, k2=3, p2=1, device='cuda', dtype=torch.float32, dropout=0.1, first_norm=True):
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels

        if first_norm==False:
            self.single_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=k1, padding=p1, bias=False, device=device, dtype=dtype),
                nn.BatchNorm2d(mid_channels, device=device, dtype=dtype), nn.ReLU(inplace=True), nn.Dropout(dropout))
        else:
            self.single_conv = nn.Sequential(
                nn.BatchNorm2d(in_channels, device=device, dtype=dtype),
                nn.Conv2d(in_channels, mid_channels, kernel_size=k1, padding=p1, bias=False, device=device, dtype=dtype),
                nn.BatchNorm2d(mid_channels, device=device, dtype=dtype), nn.ReLU(inplace=True), nn.Dropout(dropout))

    def forward(self, x):
        return self.single_conv(x)

class DoubleConv_new(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, k1=1, p1=0, k2=3, p2=1, device='cuda', dtype=torch.float32, dropout=0.1, first_norm=True):
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels

        if first_norm==False:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=k1, padding=p1, bias=False, device=device, dtype=dtype),
                nn.BatchNorm2d(mid_channels, device=device, dtype=dtype), nn.ReLU(inplace=True), nn.Dropout(dropout),
                nn.Conv2d(mid_channels, out_channels, kernel_size=k2, padding=p2, bias=False, device=device, dtype=dtype),
                nn.BatchNorm2d(out_channels, device=device, dtype=dtype), nn.ReLU(inplace=True), nn.Dropout(dropout))
        else:
            self.double_conv = nn.Sequential(
                nn.BatchNorm2d(in_channels, device=device, dtype=dtype),
                nn.Conv2d(in_channels, mid_channels, kernel_size=k1, padding=p1, bias=False, device=device, dtype=dtype),
                nn.BatchNorm2d(mid_channels, device=device, dtype=dtype), nn.ReLU(inplace=True), nn.Dropout(dropout),
                nn.Conv2d(mid_channels, out_channels, kernel_size=k2, padding=p2, bias=False, device=device, dtype=dtype),
                nn.BatchNorm2d(out_channels, device=device, dtype=dtype), nn.ReLU(inplace=True), nn.Dropout(dropout))

    def forward(self, x):
        return self.double_conv(x)

class Up_new(nn.Module):
    def __init__(self, in_channels, out_channels, k1=1, p1=0, k2=3, p2=1, device='cuda', dtype=torch.float32, dropout=0.1, first_norm=True):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2, device=device, dtype=dtype)
        self.conv = SingleConv(in_channels, out_channels, k1=k1, p1=p1, k2=k2, p2=p2, device=device, dtype=dtype, dropout=dropout, first_norm=first_norm)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv_new(nn.Module):
    def __init__(self, in_channels, out_channels, device='cuda', dtype=torch.float32):
        super(OutConv_new, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, device=device, dtype=dtype)

    def forward(self, x):
        return self.conv(x)

class improved_UNet(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, d1=32, d2=64, d3=128, d4=256, d5=512, kernel1=1, padding1=0, kernel2=3, padding2=1, device='cuda', dtype=torch.float32, dropout=0.1, first_norm=True):
        super(improved_UNet, self).__init__()

        self.doubleconv1    = DoubleConv_new(in_channels, d1, k1=kernel1, p1=padding1, k2=kernel2, p2=padding2, device=device, dtype=dtype, dropout=dropout, first_norm=first_norm)
        self.in2_layer    = SingleConv(in_channels, d1, k1=kernel1, p1=padding1, k2=kernel2, p2=padding2, device=device, dtype=dtype, dropout=dropout, first_norm=first_norm)
        self.in3_layer    = SingleConv(in_channels, d2, k1=kernel1, p1=padding1, k2=kernel2, p2=padding2, device=device, dtype=dtype, dropout=dropout, first_norm=first_norm)
        self.in4_layer    = SingleConv(in_channels, d3, k1=kernel1, p1=padding1, k2=kernel2, p2=padding2, device=device, dtype=dtype, dropout=dropout, first_norm=first_norm)
        self.max_pool     = nn.MaxPool2d(2)

        self.doubleconv2  = DoubleConv_new(2*d1, d2, k1=kernel1, p1=padding1, k2=kernel2, p2=padding2, device=device, dtype=dtype, dropout=dropout, first_norm=first_norm)
        self.doubleconv3  = DoubleConv_new(2*d2, d3, k1=kernel1, p1=padding1, k2=kernel2, p2=padding2, device=device, dtype=dtype, dropout=dropout, first_norm=first_norm)
        self.doubleconv4  = DoubleConv_new(2*d3, d4, k1=kernel1, p1=padding1, k2=kernel2, p2=padding2, device=device, dtype=dtype, dropout=dropout, first_norm=first_norm)
        self.doubleconv5  = DoubleConv_new(d4, d5, k1=kernel1, p1=padding1, k2=kernel2, p2=padding2, device=device, dtype=dtype, dropout=dropout, first_norm=first_norm)

        self.up1          = Up_new  (d5, d4, k1=kernel1, p1=padding1, k2=kernel2, p2=padding2, device=device, dtype=dtype, dropout=dropout, first_norm=first_norm)
        self.up2          = Up_new  (d4, d3, k1=kernel1, p1=padding1, k2=kernel2, p2=padding2, device=device, dtype=dtype, dropout=dropout, first_norm=first_norm)
        self.up3          = Up_new  (d3, d2, k1=kernel1, p1=padding1, k2=kernel2, p2=padding2, device=device, dtype=dtype, dropout=dropout, first_norm=first_norm)
        self.up4          = Up_new  (d2, d1, k1=kernel1, p1=padding1, k2=kernel2, p2=padding2, device=device, dtype=dtype, dropout=dropout, first_norm=first_norm)

        self.out_layer    = OutConv_new(d1, out_channels, device=device, dtype=dtype)
        self.sigmoid      = nn.Sigmoid()

    def forward(self, x1):

        num = torch.tensor(x1.shape[-1])
        num2 = int(num/2)
        num3 = int(num/4)
        num4 = int(num/8)

        x2 = transforms.Resize((num2,num2))(x1)
        x3 = transforms.Resize((num3,num3))(x1)
        x4 = transforms.Resize((num4,num4))(x1)

        x2 = self.in2_layer(x2)
        x3 = self.in3_layer(x3)
        x4 = self.in4_layer(x4)

        out1  = self.doubleconv1(x1)
        cat2 = torch.cat([x2, self.max_pool(out1)], dim=1)

        out2  = self.doubleconv2(cat2)
        cat3 = torch.cat([x3, self.max_pool(out2)], dim=1)

        out3  = self.doubleconv3(cat3)
        cat4 = torch.cat([x4, self.max_pool(out3)], dim=1)

        out4  = self.doubleconv4(cat4)
        out5  = self.doubleconv5(self.max_pool(out4))

        up_out   = self.up1(out5  , out4)
        up_out   = self.up2(up_out, out3)
        up_out   = self.up3(up_out, out2)
        up_out   = self.up4(up_out, out1)

        output = self.out_layer(up_out)
        output = self.sigmoid(output)

        return output