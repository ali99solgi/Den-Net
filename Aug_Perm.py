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

from Aug import *


def permanent_augment(images, masks, input_ch=1, size=(256,256)):

    transform1 = DualRandomHorizontalFlip(p=1)
    transform2 = DualRandomVerticalFlip(p=1)
    transform3 = DualRandomResizedCrop(scale=(0.5,1.0), ratio=(0.1,1.0), dim=input_ch, size=size)
    transform4 = DualRandomRotation(dim=input_ch)
    transform5 = DualRandomPerspective(dim=input_ch, size=size)
    transform6 = DualAdjustBrightness(dim=input_ch)
    transform7 = DualAdjustContrast(dim=input_ch)
    transform8 = DualAdjustSharpness(dim=input_ch)

    transform_list = [transform1,transform2,transform3,transform4,transform5,transform6,transform7,transform8]

    augmented_images = []
    augmented_masks = []

    for i in range(len(images)):

        selected_image = images[i] # H,W
        selected_mask = masks[i] # H,W

        augmented_set = [transform(selected_image,selected_mask) for transform in transform_list]

        image_set = []
        mask_set = []

        image_set = [selected_image.unsqueeze(0)]
        mask_set = [selected_mask.unsqueeze(0)]

        for j in range(len(augmented_set)):

          image_set.append(augmented_set[j][0].unsqueeze(0))
          mask_set.append(augmented_set[j][1].unsqueeze(0))

        augmented_images.append(torch.cat(image_set,dim=0))
        augmented_masks.append(torch.cat(mask_set,dim=0))

    new_images = torch.cat(augmented_images,dim=0)
    new_masks = torch.cat(augmented_masks,dim=0)

    return new_images , new_masks

size = (256,256)
input_ch = 1

train_images_path = '/content/DATASET/TRAIN-IMAGES'
train_masks_path = '/content/DATASET/TRAIN-MASKS'

train_images_save = '/content/DATASET/AUG-TRAIN-IMAGES/'
train_masks_save = '/content/DATASET/AUG-TRAIN-MASKS'

os.makedirs(train_images_save,exist_ok=True)
os.makedirs(train_masks_save,exist_ok=True)

ids = sorted([file for file in listdir(train_images_path) if isfile(join(train_images_path, file))])

for i in ids:

    image_name = os.path.join(train_images_path,i)
    image = cv2.imread(image_name,0)
    image = cv2.resize(image,size)
    image = transforms.ToTensor()(image)

    mask_name = os.path.join(train_masks_path,i)
    mask = cv2.imread(mask_name,0)
    mask = cv2.resize(mask,size)
    mask = transforms.ToTensor()(mask)

    image, mask = permanent_augment(image, mask, input_ch = input_ch, size = size)

    for j in range(len(image)):

        new_name = i[:-4] + '_' + str(j)+'.png'

        new_image = (image[j].to('cpu').detach().numpy())*255
        new_mask = (mask[j].to('cpu').detach().numpy())*255

        cv2.imwrite(os.path.join(train_images_save,new_name),new_image)
        cv2.imwrite(os.path.join(train_masks_save,new_name),new_mask)