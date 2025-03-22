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
from torchvision.utils import save_image
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import cv2
import matplotlib.pyplot as plt


import os

L=3
            

def save_images_to_folder(images, folder, att=False, att_map=None): # save images that are stored in a tensor of shape [b, 3, 224, 224] to folder
    invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.26862954, 1/0.26130258, 1/0.27577711 ]),
                                transforms.Normalize(mean = [ -0.48145466, -0.4578275, -0.40821073 ],
                                                     std = [ 1., 1., 1. ]),
                               ])
    images = invTrans(images)
    os.makedirs(folder, exist_ok=True)
    for i in range(images.shape[0]):
        image_path = os.path.join(folder, 'image_{}'.format(i))
        image = images[i]
        if att:
            image = image.permute(1, 2, 0)
            for j in range(att_map.shape[0]):
                att_map_layer = att_map[j]
                # print('att_map')
                fig = visualize_grid_to_grid(att_map_layer[i], image)
                fig.figure.savefig(image_path + ('{}.jpg'.format(j)), bbox_inches='tight', pad_inches=0, dpi=800)
                plt.close('all')
        else:
            save_image(image, image_path)


@torch.no_grad()
def visualize_grid_to_grid(att_map, image, grid_size=14, alpha=0.6, cls=True):
    # print(image.max(), image.min(), image[0])
    image1 = torch.clip(image, min=0, max=1)
    grid_image = image1.cpu().numpy()
    mask = att_map.reshape(grid_size, grid_size)
    # print(mask.shape)
    
    aa=mask.flatten()
    # print(aa.shape)
    bb=aa.sort()[0]
    # print(bb)
    alpha_min = 0.2
    alpha_max = 0.8
    index_min = int(alpha_min * aa.shape[0])
    index_max = int(alpha_max * aa.shape[0])
    # print(index)
    t_min = bb[index_min]
    t_max = bb[index_max]
    # print(Threshold)
    # print(mask[0])
    # mask = mask ** 3
    # for ii in range(3):
    #     mask[ii] -= 50 * (4-ii) 
    #     mask[13-ii] -= 50 * (4-ii)
    # for jj in range(14):
    #     mask[3, jj] -= 20 * (1 - np.cos(10 * jj))
    #     mask[4, jj] -= 5 * (1 - np.sin(5 * jj))
    # mask=mask.clamp(min=t_min, max=t_max)
    
    
    outmap_min = torch.min(mask)
    outmap_max = torch.max(mask)
    # print(outmap_min, outmap_max)
    mask = ((mask - outmap_min) / (outmap_max - outmap_min))
    # print(mask[0])
    # print(mask.max(), mask.min())
    mask = (mask * 255).int()
    # print(mask.max(), mask.min())
    mask = mask.cpu().numpy()
    # print(mask.shape)
    # print(grid_image.size)
    # print(grid_image.shape)
    mask = Image.fromarray(mask).resize([224, 224])
    plt.clf()
    fig = plt.figure(figsize=(5, 7))
    fig.tight_layout()
    # print(grid_image, mask)
    # print('imshow begin')
    plt.imshow(grid_image)
    # 为了保持可视化的一致性，确保 mask 的值被归一化到 [0, 1] 范围
    
    plt.imshow(mask / np.max(mask), alpha=alpha, cmap='rainbow')
    plt.axis('off')
    # plt.show()
    # print('over')

    return fig




class BatchLearner():
    def __init__(self, class_num, args, model):
        self.id_to_subgroups = {} # {image id : subsets containing this image}
        self.subgroups = []
        self.preds = []
        self.subgroup_prompts = []
        self.features = []
        self.class_to_neg_subgroups = {}
        self.max_length = args.max_length
        self.save_count = 0
        self.neg_count = 0
        self.batch_count = 0
        self.use_count = 0
        self.use_len = []
        self.args = args
        self.class_num = class_num
        self.image_ema = {} # for each class, store prompts
        self.text_ema = model.text_prompts_a.clone()
        self.ema_t = {} # for each class, store the t required for update
        self.thresholds = {'image_features': args.image_feature_threshold}
        self.retention = [LabelPropagation(args) for i in range(class_num)]
        
    def add_to_subgroups(self, subgroup):
        self.subgroups.append(subgroup)
        for index in subgroup:
            if index in self.id_to_subgroups:
                self.id_to_subgroups[index].append(len(self.subgroups) - 1)
            else:
                self.id_to_subgroups[index] = [0]
    
    def avg_entropy(self, outputs):
        score = outputs.softmax(1)
        log_score = torch.log(score)
        return -(score * log_score).sum() / score.shape[0]

    def get_subgroups(self, image_features, outputs):
        preds = outputs.argmax(dim=1)
        preds = preds.tolist()
        subsets = {}
        self.subgroups = []
        self.preds = []
        self.subgroup_logits = []
        self.id_to_subgroups = {}
        self.class_to_neg_subgroups = {}
        for i in range(len(preds)):
            if preds[i] in subsets:
                subsets[preds[i]].append(i)
            else:
                subsets[preds[i]] = [i]
        # print(subsets, 'subsets')
        for c in subsets.keys():                        
            self.add_to_subgroups(subsets[c])                        # Add the whole subset as a basic subgroup
            self.preds.append(c)
            if len(subsets[c]) > 1:                                  # find subgroups using 'image_features'
                subset_features = image_features[subsets[c]]
                affinity = subset_features @ subset_features.T
                # avg_affinity = affinity.mean()
                grouping_matrix = (affinity > self.thresholds['image_features']) * 1
                l = len(subsets[c])
                assert grouping_matrix.shape[0] == len(subsets[c])
                for i in range(l):
                    if 1 < grouping_matrix[i].sum() <= l - 1:
                        subgroup = []
                        for j in range(l):
                            if grouping_matrix[i, j] == 1:
                                subgroup.append(subsets[c][j])
                        self.add_to_subgroups(subgroup)
                        self.preds.append(c)
        print(self.subgroups, 'CLIQUES')

    def subgroups_prompt_tuning(self, model, images, optimizer, scaler, args, image_features, outputs, dart_text_ema, classnames=None):
        self.subgroup_prompts = []
        model.batch_learning = True
        for i, subgroup in enumerate(self.subgroups):
            # TODO: use augmentation for tuning
            
            if args.batch_aug:
                image_subgroup0 = images[:, subgroup]
                image_subgroup = image_subgroup0.reshape([image_subgroup0.shape[0] * image_subgroup0.shape[1], 3, 224, 224])
                if image_subgroup.shape[0] > 100:
                    # print('shape greater than 100')
                    image_subgroup = image_subgroup[:100, :, :, :]
            else:
                image_subgroup = images[0, subgroup]
                if image_subgroup.shape[0] > 100:
                    print('shape greater than 100')
                    image_subgroup = image_subgroup[:100, :, :, :]

            # reset the tunable prompt to its initial state
            with torch.no_grad():
                model.reset_Tclass_prompts()
                if args.reset_image_prompts:
                    model.reset_image_prompts()
                if args.share_prompts != 0 and args.reset_share_prompts:
                    model.reset_share_prompts()
            if args.reset_to_retention and (self.preds[i] in self.image_ema):
                model.state_dict()["image_prompts"].copy_(self.image_ema[self.preds[i]])
            old_text_prompt = model.text_prompts_a.clone()
            for j in range(args.tta_steps):
                with torch.cuda.amp.autocast():
                    # print(image_subset.shape, 'image subset')
                    aug_outputs, aug_image_features, _ = model(image_subgroup)
                    if args.batch_aug:
                        aug_outputs, selected_idx = select_confident_samples(aug_outputs, self.args.selection_p)
                    # print(aug_outputs.shape, 'aug output') [10, 200] or [6, 200]
                    avg_output = aug_outputs.mean(dim=0).unsqueeze(0)
                    avg_prob = avg_output.softmax(dim=1)
                    pred0 = avg_prob.argmax()
                    # print(avg_prob.shape)
                    pc = avg_prob.max()
                    negs = []
                    for k in range(avg_output.shape[1]):
                        if avg_prob[0, k] <= pc * 0.7:
                            negs.append(k)
                    loss = self.avg_entropy(aug_outputs)
                    entropy = loss.item()
                    if args.batch_aug:
                        aug_image_features = aug_image_features[selected_idx]
                    subgroup_feature = aug_image_features.mean(0).unsqueeze(0)
                    if self.args.save_groups and len(subgroup) >= 1:  
                        batch_path = './batch_att_r{}_visualization/batch_{}'.format(L, self.batch_count)
                        folder_path = os.path.join(batch_path, '_{}_{}'.format(classnames[pred0], self.save_count))
                        file_path = os.path.join(folder_path, 'features_before.txt')
                        image_folder_path = os.path.join(folder_path, 'images')
                        save_images_to_folder(image_subgroup, image_folder_path, args.att_map, model.att_map)
                        f = open(file_path, 'w')
                        # print(aug_image_features.shape, self.save_count)
                        image_list = aug_image_features.tolist()
                        f.write(str(image_list))
                        f.close()
                        self.save_count += 1
                    mse_loss = ((aug_image_features - subgroup_feature) ** 2).mean() * 100
                    loss = loss + mse_loss * self.args.mse_loss_weight
                
                optimizer.zero_grad()
                # compute gradient and do SGD step
                scaler.scale(loss).backward()
                # Unscales the gradients of optimizer's assigned params in-place
                scaler.step(optimizer)
                scaler.update()
            with torch.no_grad():
                aug_outputs_after, aug_image_features_after, _ = model(image_subgroup)
                avg_feature = aug_image_features_after.mean(0).unsqueeze(0)
                dis = ((aug_image_features_after - avg_feature)**2).sum(1).sqrt().unsqueeze(1)
                aug_image_features_after = aug_image_features_after - (aug_image_features_after - avg_feature) * dis
            # if self.args.save_groups and len(subgroup) >= 3:  
                # batch_path = './batch_att_r1_visualization/batch_{}'.format(self.batch_count)
                # folder_path = os.path.join(batch_path, '_{}_{}'.format(classnames[pred0], self.save_count))
                # file_path = os.path.join(folder_path, 'features_after.txt')
                # image_folder_path = os.path.join(folder_path, 'images_after')
                # save_images_to_folder(image_subgroup, image_folder_path, args.att_map, model.att_map)
                
                # f = open(file_path, 'w')
                # # print(aug_image_features_after.shape, self.save_count)
                # image_list = aug_image_features_after.tolist()
                # f.write(str(image_list))
                # f.close()



            self.subgroup_prompts.append([model.image_prompts.clone(), model.text_prompts_a.clone() - old_text_prompt])
            self.subgroup_logits.append(avg_output.clone())
            for k in negs:
                if k in self.class_to_neg_subgroups:
                    self.class_to_neg_subgroups[k].append(len(self.subgroup_prompts)-1)
                    assert len(self.subgroup_prompts) == i+1
                else:
                    self.class_to_neg_subgroups[k] = [len(self.subgroup_prompts)-1]
            with torch.no_grad():
                # subgroup_pred = aug_outputs.sum(dim=0).argmax().item()
                # print(aug_outputs.shape, 'aug output')
                subgroup_pred = aug_outputs.sum(dim=0).argmax().item()
                if self.preds[i] != subgroup_pred:
                    # print('what?')
                    self.preds[i] = subgroup_pred
                else:
                    # print('yes')
                    pass
                # TODO see which one is more accurate
                preds = aug_outputs.argmax(dim=1)
                self.update_retention([model.image_prompts.clone(), model.text_prompts_a.clone() - old_text_prompt], subgroup_pred, entropy, dart_text_ema, subgroup_feature.clone())
        model.batch_learning = False
   
    def update_retention(self, prompts, subgroup_pred, entropy, dart_text_ema, subgroup_feature):
        w_entropy = torch.exp(torch.tensor(-entropy) / self.args.retention_rate)
        
        if self.args.ensemble and self.args.learn_all:
            self.text_ema[:, subgroup_pred, :, :] = self.text_ema[:, subgroup_pred, :, :] + prompts[1][:, subgroup_pred,:,:] * self.args.text_w
        else:
            self.text_ema[subgroup_pred, :, :] = self.text_ema[subgroup_pred, :, :] + prompts[1][subgroup_pred,:,:] * self.args.text_w
        


        if subgroup_pred in self.image_ema:
            # print(entropy, 'entropy')
            t = self.ema_t[subgroup_pred]
            self.ema_t[subgroup_pred] += 1
            w = (1 / (1 + t)) * w_entropy
            self.image_ema[subgroup_pred] = self.image_ema[subgroup_pred] * (1 - w) + prompts[0] * w
        else:
            self.image_ema[subgroup_pred] = prompts[0]
            self.ema_t[subgroup_pred] = 1

        # group retention
        self.retention[subgroup_pred].forward([subgroup_feature], [prompts[0].clone()])
        
    
    @torch.no_grad()
    def update_dart_text_ema(self, dart_text_ema):
        self.neg_count = 0
        for i, subgroup in enumerate(self.subgroups):
            subgroup_pred = self.preds[i]
            # if len(subgroup) <= 1:
            #     continue
            if subgroup_pred in self.class_to_neg_subgroups:
                neg_subgroups = self.class_to_neg_subgroups[subgroup_pred]
                neg_prompt = 0
                for neg_subgroup_id in neg_subgroups:
                    logits = self.subgroup_logits[neg_subgroup_id]
                    grads = (logits * torch.log(logits)).sum() - logits.sum() * torch.log(logits)
                    # print(logits, similarity, 'logits, and similarity')
                    w0 = grads[0, subgroup_pred]
                    if w0 > 1:
                        w1 = torch.pow((1 / w0), self.args.w_pow)
                        if self.args.ensemble and self.args.learn_all:
                            neg_step = self.subgroup_prompts[neg_subgroup_id][1][:,subgroup_pred,:,:]
                        else:
                            neg_step = self.subgroup_prompts[neg_subgroup_id][1][subgroup_pred,:,:]
                            
                        # print(neg_step.shape, 'neg step') # [6, 512]
                        neg_prompt = neg_prompt + neg_step * w1
                        self.neg_count += 1
                neg_prompt = (neg_prompt / len(neg_subgroups)) * self.args.w_step
                if self.args.ensemble and self.args.learn_all:
                    prompt = neg_prompt + self.subgroup_prompts[i][1][:,subgroup_pred,:,:]
                else:
                    prompt = neg_prompt + self.subgroup_prompts[i][1][subgroup_pred,:,:]
                    
                dart_text_ema.step_one_weight("text_prompts_a", prompt, subgroup_pred, w=self.args.w_prompt)
        # print(self.neg_count)

    @torch.no_grad()
    def get_prompt(self, image_feature, image_id, args, pred=None, retention=True):
        subgroups_id = self.id_to_subgroups[image_id]
        image_prompts = []
        for k in subgroups_id:
            image_prompts.append(self.subgroup_prompts[k][0])
        assert len(image_prompts) > 0
        if retention:
            if pred in self.image_ema:
                for i in range(len(image_prompts)):
                    image_prompts[i] = image_prompts[i] * 0.9 + self.image_ema[pred] * 0.1
            if self.retention[pred].rc:
                rc = torch.cat(self.retention[pred].rc, dim=0)
                # rcp = torch.cat(self.retention[pred].rcp, dim=0)
                # print(rcp.shape)
                affinity = image_feature @ rc.T
                topk, indices = torch.topk(affinity.squeeze(), 1)
                # print(indices)
                # affinity = affinity.softmax(1).squeeze().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                # print(affinity.shape, affinity)
                # ret_prompt = torch.sum(affinity * rcp, dim=0)
                ret_prompt = self.retention[pred].rcp[indices]
                image_prompts.append(ret_prompt)
                # print(image_prompts[-1].shape, image_prompts[0].shape, 'should be same')
        image_prompts = torch.cat(image_prompts, dim=2) # TODO: use summation 
        return [image_prompts, self.text_ema]

    def print_prompt_affinity(self):
        temp = []
        for p in self.subgroup_prompts:
            temp.append(p[0].reshape([2*768]).unsqueeze(0))
        temp = torch.cat(temp, dim=0)
        temp = temp / temp.norm(dim=1, keepdim=True)
        a = temp @ temp.T
        print('subgroup prompts: ', a.shape[0], a.shape[0] ** 2 - a.sum().item())
        temp = []
        for c in self.image_ema.keys():
            temp.append(self.image_ema[c].reshape([2*768]).unsqueeze(0))
        temp = torch.cat(temp, dim=0)
        temp = temp / temp.norm(dim=1, keepdim=True)
        a = temp @ temp.T
        print('group retention prompts: ', a.shape[0], a.shape[0] ** 2 - a.sum().item())
        # for i in range(a.shape[0]):
        #     print(a[i])

def select_confident_samples(logits, top):
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * top)]
    return logits[idx], idx


class LabelPropagation:
    """Label Propagation"""

    def __init__(self, args):

        self.rc = []
        self.rcp = []
        self.limit = args.limit
        self.degree = args.degree
        self.graph_alpha = args.graph_alpha
        self.threshold = 0.8

    def forward(self, rn, rnp):
        # init
        eps = np.finfo(float).eps
        eps = torch.tensor(eps)
        eps = eps.cuda()

        if len(self.rc) + len(rn) <= self.limit:
            self.rc = self.rc + rn
            self.rcp = self.rcp + rnp
            return

        # Step1: Embedding
        emb_all = torch.cat(self.rc + rn, dim=0)
        N = emb_all.size(0)

        # Step2: Graph Construction
        # W
        emb1 = torch.unsqueeze(emb_all, 1)  # N*1*d
        emb2 = torch.unsqueeze(emb_all, 0)  # 1*N*d
        W = ((emb1 - emb2) ** 2).mean(2)  # N*N*d -> N*N
        # print('embedding before', torch.sum(W), W.shape)
        W = torch.exp(-W / 0.1)
        # print('final \'affinity\' matrix before', W)
        

        # keep top-k values
        topk, indices = torch.topk(W, 1) # orig 5
        mask = torch.zeros_like(W)
        mask = mask.scatter(1, indices, 1)
        mask = ((mask + torch.t(mask)) > 0).type(torch.float32)  # union, kNN graph
        # mask = ((mask>0)&(torch.t(mask)>0)).type(torch.float32)  # intersection, kNN graph
        W = W * mask

        # normalize
        D = W.sum(0)
        D_sqrt_inv = torch.sqrt(1.0 / (D + eps))
        D1 = torch.unsqueeze(D_sqrt_inv, 1).repeat(1, N)
        D2 = torch.unsqueeze(D_sqrt_inv, 0).repeat(N, 1)
        S = D1 * W * D2
        # S = S.cuda()

        # Step3: Label Propagation, F = (I-\alpha S)^{-1}Y
        y0 = torch.ones(N)
        y = torch.diag(y0).cuda()
        # F = torch.matmul(torch.inverse(torch.eye(N).cuda() - 0.3 * S + eps), y)
        F = torch.matmul(torch.inverse(torch.eye(N).cuda() - 0.3 * S + eps), emb_all)

        # Step4: divide groups
        F1 = torch.unsqueeze(F, 1)  # N*1*d
        F2 = torch.unsqueeze(F, 0)  # 1*N*d
        W = ((F1 - F2) ** 2).mean(2)  # N*N*d -> N*N
        # print('embedding after GNN', torch.sum(W), W.shape)
        W = torch.exp(-W / 0.1)
        ones = torch.ones(N)
        ones_diag = torch.diag(ones).cuda()
        W = W - ones_diag
        W = W.reshape(N * N)
        # print('final \'affinity\' matrix', W)
        topk, indices = torch.topk(W, 2 * (N - self.limit))
        merge = []
        affected = []
        for indice in indices:
            a = indice % N
            b = indice // N
            i = [a, b]
            i.sort()
            # print(i)
            if i not in merge:
                merge.append(i)
                affected.append(i[0])
                affected.append(i[1])
        new_rc = []
        new_rcp = []
        emb_all = self.rc + rn
        prompt_all = self.rcp + rnp
        for (i, j) in merge:
            new_feature = (emb_all[i] + emb_all[j]) / 2
            new_prompt = (prompt_all[i] + prompt_all[j]) / 2
            new_rc.append(new_feature)
            new_rcp.append(new_prompt)
        for i in range(N):
            if i not in affected:
                new_rc.append(emb_all[i])
                new_rcp.append(prompt_all[i])
        self.rc = new_rc
        self.rcp = new_rcp

        return

            




        