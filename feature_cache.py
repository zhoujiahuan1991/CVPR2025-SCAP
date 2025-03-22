import argparse

import time

from copy import deepcopy

from PIL import Image
import numpy as np

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import os


class FeatureCache():
    def __init__(self, num_class):
        self.cache = [[] for i in range(num_class)]

    def add_feature(self, class_id, feature, score, entropy):
        if len(self.cache[class_id]) < 3:
            self.cache[class_id].append((feature, score, entropy))
            self.cache[class_id].sort(key=lambda x: x[2])
        else:
            if self.cache[class_id][-1][2] >= entropy:
                self.cache[class_id].pop()
                self.cache[class_id].append((feature, score, entropy))
                self.cache[class_id].sort(key=lambda x: x[2])
        # count = 0
        # for i in range(len(self.cache)):
        #     if len(self.cache[i]) > 0:
        #         count += 1
        # print(count, len(self.cache))
    
    def get_all_feature(self):
        features = []
        for i in range(len(self.cache)):
            for j in range(len(self.cache[i])):
                features.append(self.cache[i][j][0].unsqueeze(0))
        if len(features) == 0:
            return None
        all_feature = torch.cat(features, dim=0)
        return all_feature

    def get_all_score(self):
        scores = []
        for i in range(len(self.cache)):
            for j in range(len(self.cache[i])):
                scores.append(self.cache[i][j][1].unsqueeze(0))
        if len(scores) == 0:
            return None
        all_score = torch.cat(scores, dim=0)
        return all_score
