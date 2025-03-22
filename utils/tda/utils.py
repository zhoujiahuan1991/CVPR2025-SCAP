import torch
import numpy as np
import torch.nn.functional as F
import operator 
import math

def update_cache(cache, pred, features_loss, shot_capacity, include_prob_map=False, target=None):
    with torch.no_grad():
        item = features_loss if not include_prob_map else features_loss[:2] + [features_loss[2]]
        if pred in cache:
            if len(cache[pred]) < shot_capacity:
                cache[pred].append(item)
            elif features_loss[1] < cache[pred][-1][1]:
                cache[pred][-1] = item
            cache[pred] = sorted(cache[pred], key=operator.itemgetter(1))
        else:
            cache[pred] = [item]


def update_ema_cache(cache, pred, features_loss, shot_capacity, include_prob_map=False, target=None, mode='dart', h=5000, weight=0.1, count=0):
    with torch.no_grad():
        if mode == 'fix':
            if pred in cache:
                cache[pred][0][0] = (0.8 * cache[pred][0][0] + 0.2 * features_loss[0]).clone()
                cache[pred][0][0] = cache[pred][0][0] / cache[pred][0][0].norm(dim=1, keepdim=True)
            else:
                cache[pred] = [features_loss]
        elif mode == 'dart':
            if pred in cache:
                new_decay = math.exp(-weight/h)
                cache[pred][0][0] = (new_decay * cache[pred][0][0] + (1.0 - new_decay) * features_loss[0]).clone()
                cache[pred][0][0] = cache[pred][0][0] / cache[pred][0][0].norm(dim=1, keepdim=True)
            else:
                cache[pred] = [features_loss]
        elif mode == '1/t':
            if pred in cache:
                new_decay = torch.tensor(1 / count).cuda()
                cache[pred][0][0] = ((1 - new_decay) * cache[pred][0][0] + (new_decay) * features_loss[0]).clone()
                cache[pred][0][0] = cache[pred][0][0] / cache[pred][0][0].norm(dim=1, keepdim=True)
            else:
                cache[pred] = [features_loss]


def get_noised_pos_cache(cache, scale=0.0):
    with torch.no_grad():
        noised_cache = {}
        for class_index in sorted(cache.keys()):
            noised_cache[class_index] = []
            for feature_loss in cache[class_index]:
                feature, loss = feature_loss
                noise = torch.rand(feature.shape, device=feature.device, dtype=feature.dtype)
                noise = noise * scale
                noised_feature = feature + noise
                noised_cache[class_index].append([noised_feature, loss])
                # print(feature[0, :5], noised_feature[0, :5])
    return noised_cache





def update_text_cache(cache, pred, features_loss, shot_capacity, multifeature=False, target=None, EMA=False, h=5000, weight=0.1):
    with torch.no_grad():
        if EMA:
            if pred in cache:
                new_decay = math.exp(-weight/h)
                cache[pred][0][0] = (new_decay * cache[pred][0][0] + (1.0 - new_decay) * features_loss[0]).clone()
                cache[pred][0][0] = cache[pred][0][0] / cache[pred][0][0].norm(dim=1, keepdim=True)
            else:
                cache[pred] = [features_loss]
            return
        if multifeature:
            item = features_loss # features [2, 512], loss, two-hot vector [2, 200]
        else:
            item = features_loss
        if pred in cache:
            if len(cache[pred]) < shot_capacity:
                cache[pred].append(item)
            elif features_loss[1] < cache[pred][-1][1]:
                cache[pred][-1] = item
            cache[pred] = sorted(cache[pred], key=operator.itemgetter(1))
        else:
            cache[pred] = [item]


def compute_cache_logits(image_features, cache, alpha, beta, num_classes, neg_mask_thresholds=None):
    """Compute logits using positive/negative cache."""
    with torch.no_grad():
        cache_keys = []
        cache_values = []
        mask = torch.zeros([num_classes])
        # print(mask)
        for class_index in sorted(cache.keys()):
            mask[class_index] = 1
            for item in cache[class_index]:
                cache_keys.append(item[0])
                if neg_mask_thresholds:
                    cache_values.append(item[2])
                else:
                    cache_values.append(class_index)

        cache_keys = torch.cat(cache_keys, dim=0).permute(1, 0)
        # print('cache_keys: ', cache_keys.shape) [512, x]
        if neg_mask_thresholds:
            cache_values = torch.cat(cache_values, dim=0)
            cache_values = (((cache_values > neg_mask_thresholds[0]) & (cache_values < neg_mask_thresholds[1])).type(torch.int8)).cuda().half()
            # print(torch.sum(cache_values), cache_values.shape)
        else:
            cache_values = (F.one_hot(torch.Tensor(cache_values).to(torch.int64), num_classes=num_classes)).cuda().half()
            # cache_values = torch.cat(cache_values, dim=0)

        affinity = image_features @ cache_keys
        # print(affinity.dtype)
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        return alpha * cache_logits, mask


def compute_prototype_cache_logits(image_features, cache, alpha, beta, num_classes, neg_mask_thresholds=None):
    """Compute logits using positive/negative cache."""
    with torch.no_grad():
        cache_keys = []
        cache_values = []
        mask = torch.zeros([num_classes])
        # print(mask)
        for class_index in sorted(cache.keys()):
            mask[class_index] = 1
            for item in cache[class_index]:
                cache_keys.append(item.unsqueeze(0))
                cache_values.append(class_index)

        cache_keys = torch.cat(cache_keys, dim=0).permute(1, 0)
        # print('cache_keys: ', cache_keys.shape) [512, x]
        cache_values = (F.one_hot(torch.Tensor(cache_values).to(torch.int64), num_classes=num_classes)).cuda().half()

        affinity = image_features @ cache_keys
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        return alpha * cache_logits, mask


def compute_text_cache_logits(image_features, cache, alpha, beta, num_classes, neg=False):
    with torch.no_grad():
        cache_keys = []
        cache_values = []
        for class_index in sorted(cache.keys()):
            for item in cache[class_index]:
                cache_keys.append(item[0])
                if neg:
                    cache_values.append(item[2])
                else:
                    cache_values.append(class_index)

        cache_keys = torch.cat(cache_keys, dim=0).permute(1, 0)
        if neg:
            # print(len(cache_values))
            cache_values = torch.cat(cache_values, dim=0).cuda().half()
            # print('cache_values_shape')
            # print(cache_values.shape)
            
        else:
            cache_values = (F.one_hot(torch.Tensor(cache_values).to(torch.int64), num_classes=num_classes)).cuda().half()
        
        affinity = image_features @ cache_keys
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        return alpha * cache_logits

    
def dropout(cache, rate):
    with torch.no_grad():
        for class_index in sorted(cache.keys()):
            l = len(cache[class_index])
            if l > 1:
                r = np.random.random()
                if r < rate:
                    r1 = np.random.randint(0, l)
                    cache[class_index].pop(r1)


def get_cache_features(cache):
    with torch.no_grad():
        cache_keys1 = []
        cache_values1 = []
        for class_index in sorted(cache.keys()):
            l = len(cache[class_index])
            # print(l)
            cache_values1.append(class_index)
            cache_keys1.append(cache[class_index][0][0])
            for i in range(1, l):
                cache_keys1[-1] = cache[class_index][i][0] + cache_keys1[-1] # *****?????*****
            l = torch.tensor([l], dtype=torch.float).cuda()
            cache_keys1[-1] = cache_keys1[-1] / l
            # print(len(cache[class_index]))

        cache_features = torch.cat(cache_keys1, 0)

        return cache_features, cache_values1