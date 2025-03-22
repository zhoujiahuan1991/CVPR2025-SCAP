import argparse

import time
import math

from copy import deepcopy
import sys

from PIL import Image
import numpy as np

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import operator
from sklearn import manifold
from torch.autograd import Variable

import os

import torchvision.models as models 


count_learn = 0
count_adnew = 0


def add_feature_prompt(cache, new_feature, new_prompt):
    need_new = True
    print('cache: ', len(cache))
    for i in range(len(cache)):
        if cache[i][0] @ new_feature >= 0.95:
            print(cache[i][0] @ new_feature)
            cache[i][0] = cache[i][0] * 0.8 + new_feature * 0.2
            cache[i][0] = cache[i][0] / cache[i][0].norm()
            need_new = False
    if need_new and len(cache) <= 100:
        cache.append([new_feature, new_prompt])
    
    # counting
    global count_adnew
    global count_learn
    if need_new:
        count_adnew += 1
    else:
        count_learn += 1

    print('count_adnew: ', count_adnew)
    print('count_learn: ', count_learn)


def update_feature_prompt(cache, image_feature, learned_prompt, weight, h=4000):
    for i in range(len(cache)):
        alpha = image_feature @ cache[i][0]
        learning_rate = alpha * (1 - math.exp(-weight/h))
        cache[i][1] = cache[i][1] * (1 - learning_rate) + learned_prompt * learning_rate

def update_class_prompt(cache, learned_prompt, pred):
    # print(pred)
    count = torch.bincount(pred)
    # print(count.shape, count)
    count = count.float()
    count = 1 - torch.exp(-count * 0.01)
    # print(count)
    for i in range(count.shape[0]):
        cache[i] = cache[i] * (1 - count[i]) + learned_prompt * count[i]

        # take the top 3 classes that is in the pred, update those equally
        # _ ,top_classes = torch.topk(count, 3)
        # for j in range(1):
        #     index = top_classes[j]
        #     cache[index] = cache[index] * 0.5 + learned_prompt * 0.5

def update_all_class_prompt(cache, learned_prompt):
    for i in range(len(cache)):
        cache[i] = cache[i] * 0.8 + learned_prompt * 0.2
