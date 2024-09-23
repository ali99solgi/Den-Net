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


class DualRandomResizedCrop():
    def __init__(self, scale=(0.5, 1.0), ratio=(0.5, 1.0), dim=1, size=(256,256)):
        self.scale = scale
        self.ratio = ratio
        self.dim = dim
        self.size = size

    def __call__(self, img, mask):

        i, j, h, w = T.RandomResizedCrop.get_params(img, scale=self.scale, ratio=self.ratio)

        if self.dim == 1:
            img_transformed = TF.resized_crop(img.unsqueeze(0), i, j, h, w, size=self.size, antialias=True, interpolation=T.InterpolationMode.BILINEAR).squeeze(0)
            mask_transformed = TF.resized_crop(mask.unsqueeze(0), i, j, h, w, size=self.size, antialias=True, interpolation=T.InterpolationMode.BILINEAR).squeeze(0)

        elif self.dim == 3:
            img_transformed = TF.resized_crop(img, i, j, h, w, size=self.size, antialias=True, interpolation=T.InterpolationMode.BILINEAR)
            mask_transformed = TF.resized_crop(mask, i, j, h, w, size=self.size, antialias=True, interpolation=T.InterpolationMode.BILINEAR)

        return img_transformed, mask_transformed

#-----------------------------------------------------------------------------------------

class DualRandomRotation():
    def __init__(self, degrees=(0, 180), dim=1):
        self.degrees = degrees
        self.dim = dim

    def __call__(self, img, mask):

        angle = T.RandomRotation.get_params(degrees=self.degrees)

        if self.dim == 1:
          img_rotated = TF.rotate(img=img.unsqueeze(0), angle=angle, interpolation=T.InterpolationMode.BILINEAR).squeeze(0)
          mask_rotated = TF.rotate(img=mask.unsqueeze(0), angle=angle, interpolation=T.InterpolationMode.BILINEAR).squeeze(0)

        if self.dim == 3:
          img_rotated = TF.rotate(img=img, angle=angle, interpolation=T.InterpolationMode.BILINEAR)
          mask_rotated = TF.rotate(img=mask, angle=angle, interpolation=T.InterpolationMode.BILINEAR)

        return img_rotated , mask_rotated

#-----------------------------------------------------------------------------------------

class DualRandomPerspective():
    def __init__(self, distortion_scale=0.3, p=1.0, dim=1, size=(256,256)):
        self.distortion_scale = distortion_scale
        self.dim = dim
        self.size = size

    def __call__(self, img, mask):

        points = T.RandomPerspective.get_params(self.size[0],self.size[1],distortion_scale=self.distortion_scale)

        if self.dim == 1:
            img_transformed = TF.perspective(img=img.unsqueeze(0),startpoints=points[0],endpoints=points[1],interpolation=T.InterpolationMode.BILINEAR).squeeze(0)
            mask_transformed = TF.perspective(img=mask.unsqueeze(0),startpoints=points[0],endpoints=points[1],interpolation=T.InterpolationMode.BILINEAR).squeeze(0)

        elif self.dim == 3:
            img_transformed = TF.perspective(img=img,startpoints=points[0],endpoints=points[1],interpolation=T.InterpolationMode.BILINEAR)
            mask_transformed = TF.perspective(img=mask,startpoints=points[0],endpoints=points[1],interpolation=T.InterpolationMode.BILINEAR)

        return img_transformed, mask_transformed

#-----------------------------------------------------------------------------------------

class DualAdjustBrightness():
    def __init__(self, low=1, high=2, dim=1):
        self.low = low
        self.high = high
        self.dim = dim

    def __call__(self, img, mask):

        brightness_factor = float(np.random.uniform(low=self.low,high=self.high,size=1)[0])

        if self.dim == 1:
            img_transformed = TF.adjust_brightness(img.unsqueeze(0),brightness_factor).squeeze(0)

        elif self.dim == 3:
            img_transformed = TF.adjust_brightness(img,brightness_factor)

        return img_transformed, mask

#-----------------------------------------------------------------------------------------

class DualAdjustContrast():
    def __init__(self, low=1, high=1.5, dim=1):
        self.low = low
        self.high = high
        self.dim = dim

    def __call__(self, img, mask):

        contrast_factor = float(np.random.uniform(low=self.low,high=self.high,size=1)[0])

        if self.dim == 1:
            img_transformed = TF.adjust_contrast(img.unsqueeze(0),contrast_factor).squeeze(0)

        elif self.dim == 3:
            img_transformed = TF.adjust_contrast(img,contrast_factor)

        return img_transformed, mask

#-----------------------------------------------------------------------------------------

class DualAdjustSharpness():
    def __init__(self, low=1.5, high=3, dim=1):
        self.low = low
        self.high = high
        self.dim = dim

    def __call__(self, img, mask):

        sharpness_factor = float(np.random.uniform(low=self.low,high=self.high,size=1)[0])

        if self.dim == 1:
            img_transformed = TF.adjust_sharpness(img.unsqueeze(0),sharpness_factor).squeeze(0)

        elif self.dim == 3:
            img_transformed = TF.adjust_sharpness(img,sharpness_factor)

        return img_transformed, mask

#-----------------------------------------------------------------------------------------

class DualRandomHorizontalFlip():
    def __init__(self, p=1):
        self.p = p

    def __call__(self, img, mask):

        transform = T.RandomHorizontalFlip(p=self.p)

        img_transformed = transform(img)
        mask_transformed = transform(mask)

        return img_transformed, mask_transformed

#-----------------------------------------------------------------------------------------

class DualRandomVerticalFlip():
    def __init__(self, p=1):
        self.p = p

    def __call__(self, img, mask):

        transform = T.RandomVerticalFlip(p=self.p)

        img_transformed = transform(img)
        mask_transformed = transform(mask)

        return img_transformed, mask_transformed

#-----------------------------------------------------------------------------------------

def augment(images, masks, input_ch=1, size=(256,256)):

    transform1 = DualRandomHorizontalFlip(p=1)
    transform2 = DualRandomVerticalFlip(p=1)
    transform3 = DualRandomResizedCrop(scale=(0.4,0.6), ratio=(0.4,0.6), dim=input_ch, size=size)
    transform4 = DualRandomRotation(dim=input_ch)
    transform5 = DualRandomPerspective(dim=input_ch, size=size)
    transform6 = DualAdjustBrightness(dim=input_ch, low=1.05, high=1.2)
    transform7 = DualAdjustContrast(dim=input_ch, low=1.1, high=1.3)
    transform8 = DualAdjustSharpness(dim=input_ch, low=2.5, high=3)

    transform_list = [transform1,transform2,transform3,transform4,transform5,transform6,transform7,transform8]
    #transform_list = [transform1,transform2,transform3,transform4,transform6,transform7,transform8]
    #transform_list = [transform3,transform6,transform7,transform8]

    augmented_images = []
    augmented_masks = []

    for i in range(len(images)):

        selected_image = images[i] # H,W
        selected_mask = masks[i] # H,W
        image_set = []
        mask_set = []

        number = np.random.randint(-1,8)
        #number = np.random.randint(-1,4)

        if number != -1:
            transform = transform_list[number]
            augmented_set = transform(selected_image,selected_mask)
            image_set = augmented_set[0].unsqueeze(0)
            mask_set = augmented_set[1].unsqueeze(0)
        else:
            image_set = selected_image.unsqueeze(0)
            mask_set = selected_mask.unsqueeze(0)

        augmented_images.append(image_set)
        augmented_masks.append(mask_set)

    new_images = torch.cat(augmented_images,dim=0)
    new_masks = torch.cat(augmented_masks,dim=0)

    return new_images , new_masks
