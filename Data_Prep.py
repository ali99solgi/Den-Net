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
import sys

from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from torch import optim , Tensor
from tqdm import tqdm
from os import listdir
from os.path import isfile, join
from pathlib import Path
from copy import deepcopy
from PIL import Image

from Utils import *
from Loss_Functions import *
from Aug import *
from U_Net import *

#------------------------------------------------------------------------------------------------------------------------------------

train_range_list = [i for i in range(1,18)]

train_range_list.remove(6)

train_range_list.remove(5)
train_range_list.remove(11)
train_range_list.remove(12)
train_range_list.remove(13)

for num in train_range_list:

    src_img_path = '/content/PROCESSED/{NUM} PROCESSED IMAGE'.format(NUM = num)
    src_mask_path = '/content/PROCESSED/{NUM} PROCESSED MASK'.format(NUM = num)

    dst_img_path = f'/content/DATASET/TRAIN-IMAGES'
    dst_mask_path = f'/content/DATASET/TRAIN-MASKS'

    shutil.copytree(src_img_path, dst_img_path, dirs_exist_ok = True)
    shutil.copytree(src_mask_path, dst_mask_path, dirs_exist_ok = True)

val_range_list = [6]

for num in val_range_list:

    src_img_path = '/content/PROCESSED/{NUM} PROCESSED IMAGE'.format(NUM = num)
    src_mask_path = '/content/PROCESSED/{NUM} PROCESSED MASK'.format(NUM = num)
    src_label_path = '/content/PROCESSED/{NUM} PROCESSED LABEL'.format(NUM = num)

    dst_img_path = f'/content/DATASET/VAL-IMAGES'
    dst_mask_path = f'/content/DATASET/VAL-MASKS'
    dst_label_path = f'/content/DATASET/VAL-LABELS'

    shutil.copytree(src_img_path, dst_img_path, dirs_exist_ok = True)
    shutil.copytree(src_mask_path, dst_mask_path, dirs_exist_ok = True)
    shutil.copytree(src_label_path, dst_label_path, dirs_exist_ok = True)

test_range_list = [5,11,12,13]

for num in test_range_list:

    src_img_path = '/content/PROCESSED/{NUM} PROCESSED IMAGE'.format(NUM = num)
    src_mask_path = '/content/PROCESSED/{NUM} PROCESSED MASK'.format(NUM = num)
    src_label_path = '/content/PROCESSED/{NUM} PROCESSED LABEL'.format(NUM = num)

    dst_img_path = f'/content/DATASET/TEST-IMAGES'
    dst_mask_path = f'/content/DATASET/TEST-MASKS'
    dst_label_path = f'/content/DATASET/TEST-LABELS'

    shutil.copytree(src_img_path, dst_img_path, dirs_exist_ok = True)
    shutil.copytree(src_mask_path, dst_mask_path, dirs_exist_ok = True)
    shutil.copytree(src_label_path, dst_label_path, dirs_exist_ok = True)
    
#------------------------------------------------------------------------------------------------------------------------------------

train_range_list = [i for i in range(1,18)]

train_range_list.remove(6)

train_range_list.remove(4)
train_range_list.remove(5)
train_range_list.remove(12)
train_range_list.remove(13)

for num in train_range_list:

    src_img_path = '/content/PROCESSED/{NUM} PROCESSED IMAGE'.format(NUM = num)
    src_mask_path = '/content/PROCESSED/{NUM} PROCESSED MASK'.format(NUM = num)

    dst_img_path = f'/content/NEW_DATASET_1/TRAIN-IMAGES'
    dst_mask_path = f'/content/NEW_DATASET_1/TRAIN-MASKS'

    shutil.copytree(src_img_path, dst_img_path, dirs_exist_ok = True)
    shutil.copytree(src_mask_path, dst_mask_path, dirs_exist_ok = True)

val_range_list = [6]

for num in val_range_list:

    src_img_path = '/content/PROCESSED/{NUM} PROCESSED IMAGE'.format(NUM = num)
    src_mask_path = '/content/PROCESSED/{NUM} PROCESSED MASK'.format(NUM = num)
    src_label_path = '/content/PROCESSED/{NUM} PROCESSED LABEL'.format(NUM = num)

    dst_img_path = f'/content/NEW_DATASET_1/VAL-IMAGES'
    dst_mask_path = f'/content/NEW_DATASET_1/VAL-MASKS'
    dst_label_path = f'/content/NEW_DATASET_1/VAL-LABELS'

    shutil.copytree(src_img_path, dst_img_path, dirs_exist_ok = True)
    shutil.copytree(src_mask_path, dst_mask_path, dirs_exist_ok = True)
    shutil.copytree(src_label_path, dst_label_path, dirs_exist_ok = True)

test_range_list = [4,5,12,13]

for num in test_range_list:

    src_img_path = '/content/PROCESSED/{NUM} PROCESSED IMAGE'.format(NUM = num)
    src_mask_path = '/content/PROCESSED/{NUM} PROCESSED MASK'.format(NUM = num)
    src_label_path = '/content/PROCESSED/{NUM} PROCESSED LABEL'.format(NUM = num)

    dst_img_path = f'/content/NEW_DATASET_1/TEST-IMAGES'
    dst_mask_path = f'/content/NEW_DATASET_1/TEST-MASKS'
    dst_label_path = f'/content/NEW_DATASET_1/TEST-LABELS'

    shutil.copytree(src_img_path, dst_img_path, dirs_exist_ok = True)
    shutil.copytree(src_mask_path, dst_mask_path, dirs_exist_ok = True)
    shutil.copytree(src_label_path, dst_label_path, dirs_exist_ok = True)
    
#------------------------------------------------------------------------------------------------------------------------------------

train_range_list = [i for i in range(1,18)]

train_range_list.remove(6)

train_range_list.remove(5)
train_range_list.remove(8)
train_range_list.remove(12)

for num in train_range_list:

    src_img_path = '/content/PROCESSED/{NUM} PROCESSED IMAGE'.format(NUM = num)
    src_mask_path = '/content/PROCESSED/{NUM} PROCESSED MASK'.format(NUM = num)

    dst_img_path = f'/content/NEW_DATASET_2/TRAIN-IMAGES'
    dst_mask_path = f'/content/NEW_DATASET_2/TRAIN-MASKS'

    shutil.copytree(src_img_path, dst_img_path, dirs_exist_ok = True)
    shutil.copytree(src_mask_path, dst_mask_path, dirs_exist_ok = True)

val_range_list = [6]

for num in val_range_list:

    src_img_path = '/content/PROCESSED/{NUM} PROCESSED IMAGE'.format(NUM = num)
    src_mask_path = '/content/PROCESSED/{NUM} PROCESSED MASK'.format(NUM = num)
    src_label_path = '/content/PROCESSED/{NUM} PROCESSED LABEL'.format(NUM = num)

    dst_img_path = f'/content/NEW_DATASET_2/VAL-IMAGES'
    dst_mask_path = f'/content/NEW_DATASET_2/VAL-MASKS'
    dst_label_path = f'/content/NEW_DATASET_2/VAL-LABELS'

    shutil.copytree(src_img_path, dst_img_path, dirs_exist_ok = True)
    shutil.copytree(src_mask_path, dst_mask_path, dirs_exist_ok = True)
    shutil.copytree(src_label_path, dst_label_path, dirs_exist_ok = True)

test_range_list = [5,8,12]

for num in test_range_list:

    src_img_path = '/content/PROCESSED/{NUM} PROCESSED IMAGE'.format(NUM = num)
    src_mask_path = '/content/PROCESSED/{NUM} PROCESSED MASK'.format(NUM = num)
    src_label_path = '/content/PROCESSED/{NUM} PROCESSED LABEL'.format(NUM = num)

    dst_img_path = f'/content/NEW_DATASET_2/TEST-IMAGES'
    dst_mask_path = f'/content/NEW_DATASET_2/TEST-MASKS'
    dst_label_path = f'/content/NEW_DATASET_2/TEST-LABELS'

    shutil.copytree(src_img_path, dst_img_path, dirs_exist_ok = True)
    shutil.copytree(src_mask_path, dst_mask_path, dirs_exist_ok = True)
    shutil.copytree(src_label_path, dst_label_path, dirs_exist_ok = True)

#------------------------------------------------------------------------------------------------------------------------------------

train_range_list = [i for i in range(1,18)]

train_range_list.remove(6)

train_range_list.remove(1)
train_range_list.remove(2)
train_range_list.remove(3)

for num in train_range_list:

    src_img_path = '/content/PROCESSED/{NUM} PROCESSED IMAGE'.format(NUM = num)
    src_mask_path = '/content/PROCESSED/{NUM} PROCESSED MASK'.format(NUM = num)

    dst_img_path = f'/content/NEW_DATASET_3/TRAIN-IMAGES'
    dst_mask_path = f'/content/NEW_DATASET_3/TRAIN-MASKS'

    shutil.copytree(src_img_path, dst_img_path, dirs_exist_ok = True)
    shutil.copytree(src_mask_path, dst_mask_path, dirs_exist_ok = True)

val_range_list = [6]

for num in val_range_list:

    src_img_path = '/content/PROCESSED/{NUM} PROCESSED IMAGE'.format(NUM = num)
    src_mask_path = '/content/PROCESSED/{NUM} PROCESSED MASK'.format(NUM = num)
    src_label_path = '/content/PROCESSED/{NUM} PROCESSED LABEL'.format(NUM = num)

    dst_img_path = f'/content/NEW_DATASET_3/VAL-IMAGES'
    dst_mask_path = f'/content/NEW_DATASET_3/VAL-MASKS'
    dst_label_path = f'/content/NEW_DATASET_3/VAL-LABELS'

    shutil.copytree(src_img_path, dst_img_path, dirs_exist_ok = True)
    shutil.copytree(src_mask_path, dst_mask_path, dirs_exist_ok = True)
    shutil.copytree(src_label_path, dst_label_path, dirs_exist_ok = True)

test_range_list = [1,2,3]

for num in test_range_list:

    src_img_path = '/content/PROCESSED/{NUM} PROCESSED IMAGE'.format(NUM = num)
    src_mask_path = '/content/PROCESSED/{NUM} PROCESSED MASK'.format(NUM = num)
    src_label_path = '/content/PROCESSED/{NUM} PROCESSED LABEL'.format(NUM = num)

    dst_img_path = f'/content/NEW_DATASET_3/TEST-IMAGES'
    dst_mask_path = f'/content/NEW_DATASET_3/TEST-MASKS'
    dst_label_path = f'/content/NEW_DATASET_3/TEST-LABELS'

    shutil.copytree(src_img_path, dst_img_path, dirs_exist_ok = True)
    shutil.copytree(src_mask_path, dst_mask_path, dirs_exist_ok = True)
    shutil.copytree(src_label_path, dst_label_path, dirs_exist_ok = True)
    
#------------------------------------------------------------------------------------------------------------------------------------

def num_Show(temp_dataset):

  train_num = f"/content/{temp_dataset}/TRAIN-IMAGES"
  val_num = f"/content/{temp_dataset}/VAL-IMAGES"
  test_num = f"/content/{temp_dataset}/TEST-IMAGES"

  train_samples = len(os.listdir(train_num))
  val_samples = len(os.listdir(val_num))
  test_samples = len(os.listdir(test_num))

  print('train samples:',train_samples,'---','val samples:',val_samples,'---')
  print('test_ samples:',test_samples,)
  print('-------------------------------------------------------------------------------------------')

num_Show('DATASET')
num_Show('NEW_DATASET_1')
num_Show('NEW_DATASET_2')
num_Show('NEW_DATASET_3')