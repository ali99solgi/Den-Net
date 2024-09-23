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


class dataset(Dataset):

    def __init__(self, images_path: str, masks_path: str, size: tuple = (256,256), input_ch: int = 1, th: int = 225, normalize: bool = False):
        self.images_path = Path(images_path)
        self.masks_path = Path(masks_path)
        self.size = size
        self.input_ch = input_ch
        self.th = th
        self.normalize = normalize
        self.ids = sorted([file for file in listdir(self.images_path) if isfile(join(self.images_path, file))])
        self.mask_ids = sorted([file for file in listdir(self.masks_path) if isfile(join(self.masks_path, file))])
        assert self.ids == self.mask_ids , 'Corresponding images and masks should have similar names'

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):

        image_name = os.path.join(self.images_path,self.ids[idx])

        if self.input_ch == 3:
            image = cv2.imread(image_name)
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            image = cv2.resize(image,self.size)
            image = transforms.ToTensor()(image)

        else:
            image = cv2.imread(image_name,0)
            image = cv2.resize(image,self.size)
            image = transforms.ToTensor()(image)

        if self.normalize == True:
            m, s = torch.mean(image, axis=(1, 2)), torch.std(image, axis=(1, 2))
            preprocess = transforms.Normalize(mean=m, std=s)
            image = preprocess(image)
        #-----------------------------------------------------------------------

        mask_name = os.path.join(self.masks_path,self.ids[idx])

        mask = cv2.imread(mask_name,0)
        mask = cv2.resize(mask,self.size)
        mask = transforms.ToTensor()(mask)
        mask = (mask >= 0.5).float()

        #-----------------------------------------------------------------------

        sample = {'images': image, 'masks': mask}

        return sample

class special_dataset(Dataset):

    def __init__(self, images_path: str, masks_path: str, labels_path: str, size: tuple = (256,256), input_ch: int = 1, th: int = 225, normalize: bool = False):
        self.images_path = Path(images_path)
        self.masks_path = Path(masks_path)
        self.labels_path = Path(labels_path)
        self.size = size
        self.input_ch = input_ch
        self.th = th
        self.normalize = normalize
        self.ids = sorted([file for file in listdir(self.images_path) if isfile(join(self.images_path, file))])
        self.mask_ids = sorted([file for file in listdir(self.masks_path) if isfile(join(self.masks_path, file))])
        self.label_ids = sorted([file for file in listdir(self.labels_path) if isfile(join(self.labels_path, file))])
        assert self.ids == self.mask_ids and self.ids == self.label_ids , 'Corresponding images and masks should have similar names'

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):

        image_name = os.path.join(self.images_path,self.ids[idx])

        if self.input_ch == 3:
            image_view = cv2.imread(image_name)
            image_view = cv2.cvtColor(image_view,cv2.COLOR_BGR2RGB)
            image_view = cv2.resize(image_view,self.size)
            image_view = transforms.ToTensor()(image_view)
            image = image_view

        else:
            image_view = cv2.imread(image_name,0)
            image_view = cv2.resize(image_view,self.size)
            image_view = transforms.ToTensor()(image_view)
            image = image_view

            if self.normalize == True:

                m, s = torch.mean(image_view, axis=(1, 2)), torch.std(image_view, axis=(1, 2))
                preprocess = transforms.Normalize(mean=m, std=s)
                image = preprocess(image_view)

        #-----------------------------------------------------------------------

        mask_name = os.path.join(self.masks_path,self.ids[idx])

        mask = cv2.imread(mask_name,0)
        mask = cv2.resize(mask,self.size)
        mask = transforms.ToTensor()(mask)
        mask = (mask >= 0.5).float()

        #-----------------------------------------------------------------------

        label_name = os.path.join(self.labels_path,self.ids[idx])

        label = cv2.imread(label_name)
        label = cv2.cvtColor(label,cv2.COLOR_BGR2RGB)
        label = cv2.resize(label,self.size)
        label = torch.tensor(label,dtype=torch.float32)/255

        #-----------------------------------------------------------------------

        sample = {'images': image, 'masks': mask, 'labels':label, "images_view":image_view}

        return sample, self.ids[idx]


#------------------------------------------------------------------------------------------------

def color(image,thresh,R,G,B):

  color_img = image.repeat(3,1,1)
  color_img = (color_img.transpose(0,1)).transpose(1,2)

  for j in range(thresh.shape[0]):
    for k in range(thresh.shape[1]):
      if thresh[j,k] == 1:
        color_img[j,k,0] = R
        color_img[j,k,1] = G
        color_img[j,k,2] = B

  return color_img

def calculate_metrics(pred, target, epsilon = 1e-12):

    # Check if single image or batch is provided
    if pred.dim() == 3:  # Single image (1, W, H)
        pred = pred.unsqueeze(0)  # Add batch dimension (1, 1, W, H)
        target = target.unsqueeze(0)  # Add batch dimension (1, 1, W, H)

    #---------------------------------------------------------------------------

    # Flatten predictions and targets
    pred = pred.flatten(1).float()
    target = target.flatten(1).float()

    # convert integers to binary values
    pred = pred.bool()
    target = target.bool()

    # True Positives, False Positives, True Negatives, False Negatives
    TP = torch.sum((pred & target ).float(),dim=1)
    FP = torch.sum((pred & (~target)).float(),dim=1)
    FN = torch.sum(((~pred) & target).float(),dim=1)
    TN = torch.sum(((~pred) & (~target)).float(),dim=1)

    #---------------------------------------------------------------------------

    iou = TP/(TP + FP + FN + epsilon)

    dice = 2*TP/(2*TP + FP + FN + epsilon)

    sensitivity = TP / (TP + FN + epsilon)

    specificity = TN / (TN + FP + epsilon)

    precision = TP / (TP + FP + epsilon)

    recall = sensitivity
    F1_score = (2 * precision * recall) / (precision + recall + epsilon)

    return torch.sum(iou), torch.sum(dice), torch.sum(sensitivity), torch.sum(specificity), torch.sum(precision), torch.sum(F1_score)

def calculate_metrics_for_test(pred, target, epsilon = 1e-12):

    # Check if single image or batch is provided
    if pred.dim() == 3:  # Single image (1, W, H)
        pred = pred.unsqueeze(0)  # Add batch dimension (1, 1, W, H)
        target = target.unsqueeze(0)  # Add batch dimension (1, 1, W, H)

    #---------------------------------------------------------------------------

    # Flatten predictions and targets
    pred = pred.flatten(1).float()
    target = target.flatten(1).float()

    # Calculate MAE
    mae = torch.sum(torch.mean(torch.abs(pred - target),dim=1))

    # convert integers to binary values
    pred = pred.bool()
    target = target.bool()

    # True Positives, False Positives, True Negatives, False Negatives
    TP = torch.sum((pred & target ).float(),dim=1)
    FP = torch.sum((pred & (~target)).float(),dim=1)
    FN = torch.sum(((~pred) & target).float(),dim=1)
    TN = torch.sum(((~pred) & (~target)).float(),dim=1)

    #---------------------------------------------------------------------------

    iou = TP/(TP + FP + FN + epsilon)

    dice = 2*TP/(2*TP + FP + FN + epsilon)

    sensitivity = TP / (TP + FN + epsilon)

    specificity = TN / (TN + FP + epsilon)

    precision = TP / (TP + FP + epsilon)

    recall = sensitivity
    F1_score = (2 * precision * recall) / (precision + recall + epsilon)

    return torch.sum(iou), torch.sum(dice), torch.sum(sensitivity), torch.sum(specificity), torch.sum(precision), torch.sum(F1_score), mae
