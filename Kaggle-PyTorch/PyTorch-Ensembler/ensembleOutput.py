from __future__ import print_function, absolute_import

import errno
import time

import matplotlib
import torch.nn as nn
import torch.nn.init as init

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import datetime
import pandas as pd
from torch.utils.data import TensorDataset

import torch.nn.parallel
from sklearn.utils import shuffle
from torch.autograd import Variable
from torch.utils.data import TensorDataset
from torchvision.transforms import *
import numpy as np

import nnmodels as nnmodels

from os import listdir
import sys

from utils import *

#'/home/adodd202/train.json'

#/Kaggle-Statoil-Challenge/Kaggle-PyTorch/PyTorch-Ensembler/log/statoil/IceResNet/pth

input_folder = '/home/adodd202/Kaggle-Statoil-Challenge/Kaggle-PyTorch/PyTorch-Ensembler/log/statoil/IceResNet/pth'
best_base = '/home/adodd202/Kaggle-Statoil-Challenge/Kaggle-PyTorch/PyTorch-Ensembler/log/statoil/IceResNet/pth/0.164028_0.163341_submission.csv'
output_path = '/home/adodd202/Kaggle-Statoil-Challenge/Kaggle-PyTorch/PyTorch-Ensembler/log/statoil/IceResNet/pth/minMaxBaseStack.csv'
MinMaxBestBaseStacking(input_folder, best_base, output_path)

#output_path = '/home/adodd202/Kaggle-Statoil-Challenge/Kaggle-PyTorch/PyTorch-Ensembler/log/statoil/IceResNet/pth/ensembleVer2.csv'
#ensembleVer2(input_folder, output_path)