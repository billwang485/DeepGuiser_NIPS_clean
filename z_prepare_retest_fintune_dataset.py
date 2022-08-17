import os
from shutil import copy
from matplotlib import collections
import torch
dir = ['./predictor/finetune_dataset/Apr 24 06 06 58',
    './predictor/finetune_dataset/Apr 24 06 09 24']


import os
from pickle import TRUE
import sys
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import time
import re
from single_model import NASNetwork as Network
from nat_learner_twin import Transformer
import random
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter

def main():
    torch.save(dir, 'dirs')


    target_dirs = []
    surrogate_dirs = []
    all_data = []

    for i, path_ in enumerate(dir):
        
        for j in range(1000):
            data_point = OrderedDict()
            if os.path.exists(os.path.join(path_, str(j), 'target.pt')) and os.path.exists(os.path.join(path_, str(j), 'target_baseline.pt')) and os.path.exists(os.path.join(path_, str(j), 'surrogate_model.pt')):
                data_point['target_dirs'] = os.path.join(path_, str(j))
                data_point['surrogate_path'] = os.path.join(path_, str(j), 'surrogate_model.pt')
                data_point['available'] = True
                all_data.append(data_point)
    torch.save(all_data, 'retest_archs')

if __name__ == '__main__':
    main()