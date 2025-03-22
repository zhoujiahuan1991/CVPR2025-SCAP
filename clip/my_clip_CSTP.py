
import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from clip import load, tokenize
from .simple_tokenizer import SimpleTokenizer as _Tokenizer
from data.imagnet_prompts import imagenet_classes
from data.fewshot_datasets import fewshot_datasets
from data.cls_to_names import *
import os
from transformers import BlipProcessor, BlipModel, AutoProcessor



import numpy as np

# from utils.prompt_pool import PromptPool

from torch.utils.checkpoint import checkpoint

_tokenizer = _Tokenizer()

DOWNLOAD_ROOT='../~/.cache/clip'


class ClipTestTimeTuning(nn.Module):
    def __init__(
        self, args, device, classnames,
        criterion='cosine', arch="ViT-L/14"
    ):
        super(ClipTestTimeTuning, self).__init__()
        clip, _, preprocess = load(arch, device=device, download_root=DOWNLOAD_ROOT)
        self.preprocess = preprocess
        self.clip = clip
        self.image_encoder = clip.visual
        # self.text_encoder = TextEncoder(clip)
        self.logit_scale = clip.logit_scale.data
        self.criterion = criterion
        self.args = args
        self.train_mode = True
        self.xs = []
        self.p_lens = []
        self.att_map = None
        self.simam = SimamModule()
        self.batch_learning = False
        self.dummy = torch.ones(1, dtype=torch.float32, requires_grad=True)
        if self.args.image_prompts:
            self.image_prompt_layers = self.args.image_prompt_layer
            if self.args.prefix_tuning:
                if self.args.arch == "ViT-L/14":
                    self.image_prompts = torch.empty(
                    (1, len(self.image_prompt_layers), 2, 1024), 
                    dtype=self.clip.dtype,
                    device="cuda").uniform_(-1, 1).requires_grad_(True)
                    self.use_sum = args.use_sum
                else:
                    
                    self.image_prompts = torch.empty(
                        # TODO
                        ### per image uses the same prompt
                        (1, len(self.image_prompt_layers), 2, 768), 
                        ### per image uses different prompt
                        # (self.args.batch_size, len(self.image_prompt_layers), 2, 768), 
                        dtype=self.clip.dtype,
                        device="cuda").uniform_(-1, 1).requires_grad_(True)
                    self.use_sum = args.use_sum
            else:
                self.image_prompts = torch.empty(
                    # TODO
                    # (1, 1, 768), 
                    (self.args.batch_size, 1, 768), 
                    dtype=self.clip.dtype,
                    device="cuda").uniform_(-1, 1).requires_grad_(True)
            self.image_prompts = nn.Parameter(self.image_prompts)
        
    @property
    def dtype(self):
        return self.image_encoder.conv1.weight.dtype
    
    def reset(self):
        self.reset_Tclass_prompts()
        # print(model.image_prompts.requires_grad)
        if self.args.image_prompts and self.args.reset_image_prompts is True:
            # print(model.image_prompts.requires_grad)
            self.reset_image_prompts()
            # print(model.image_prompts.requires_grad)

    def reset_share_prompts(self):
        self.share_prompts.uniform_(-1, 1).requires_grad_(True)
    
    def reset_image_prompts(self):
        # print("reset image prompts")
        # print(self.image_prompts[0:10])
        self.image_prompts.uniform_(-1, 1).requires_grad_(True)
        
    def reset_Tclass_prompts(self):
        # print("reset Tclass prompts")
        # TODO
        if self.args.CSTP == 1:
            if self.args.learnable_text == "a":
                if self.args.ensemble and self.args.learn_all:
                    self.text_prompts_a.copy_(self.text_prompts_a_inits)
                    # for i, text_prompts_a in enumerate(self.text_prompts_as):
                    #     text_prompts_a.copy_(self.text_prompts_a_inits[i])
                else:
                    self.text_prompts_a.copy_(self.text_prompts_a_init)
            elif self.args.learnable_text == "a+cls":
                self.text_prompts_a.copy_(self.text_prompts_a_init)
                self.text_prompts_class.copy_(self.text_prompts_class_init)
            elif self.args.learnable_text == "S+a+cls+E":
                self.text_prompts_S.copy_(self.text_prompts_S_init)
                self.text_prompts_a.copy_(self.text_prompts_a_init)
                self.text_prompts_class.copy_(self.text_prompts_class_init)
                self.text_prompts_E.copy_(self.text_prompts_E_init)
            elif self.args.learnable_text == "all":
                self.text_prompts.copy_(self.text_prompts_init)
        elif self.args.CSTP == 2:
            if self.args.learnable_text == "a":
                self.CSTP_bvector.copy_(self.text_prompts_a_init)
                text_prompts_a = torch.matmul(self.CSTP_weight, self.CSTP_bvector.reshape(self.args.CSTP_N, -1))
                text_prompts_a = text_prompts_a.reshape(len(self.classnames), 4, -1)
                self.text_prompts_a.copy_(text_prompts_a)
        return



    def init_text_prompts(self, classnames):        
        # 如果每个类使用一个单独的文本prompt
        if not self.args.ensemble:
            self.classnames = classnames
            # print(self.args.ctx_init)
            # print(self.args.n_ctx)
            ctx_init = self.args.ctx_init.replace("_", " ")
            ctx_init = self.args.ctx_init.replace("-", " ")
            # print(ctx_init)
            # input()
            # x = ['This is a photo of a '+classname for classname in classnames] 
            # p_len = 6
            x = [ctx_init.replace("CLS", classname) for classname in classnames] 
            # print(x) 
            # input()
            # x = ['a photo of a'+classname for classname in classnames] 
            p_len = self.args.n_ctx
            # x = ["X" * self.args.text_plen] + x
            # x = ["X " + p for p in x]
            # x = ["X " * self.args.text_plen + p for p in x]
            x = tokenize(x) # [class_num, n_ctx]
            self.tokenized_text = x.detach().clone()
            x = x.to("cuda")
            self.text_prompts_token = x.detach().clone()
            x = self.clip.token_embedding(x).type(self.dtype)  # [class_num, n_ctx, d_model]
            # TODO
            if self.args.learnable_text == "a":
                self.text_prompts_begin = x[:,0,:].detach().clone().unsqueeze(1)
                self.text_prompts_a = x[:,1:p_len+1,:].detach().clone()
                self.text_prompts_a = nn.Parameter(self.text_prompts_a)
                self.text_prompts_a_init = x[:,1:p_len+1,:].detach().clone()
                self.text_prompts_class = x[:,p_len+1,:].detach().clone().unsqueeze(1)
                self.text_prompts_end = x[:,p_len+2:,:].detach().clone()
            elif self.args.learnable_text == "a+cls":
                self.text_prompts_begin = x[:,0,:].detach().clone().unsqueeze(1)
                self.text_prompts_a = x[:,1:5,:].detach().clone()
                self.text_prompts_a = nn.Parameter(self.text_prompts_a)
                self.text_prompts_a_init = x[:,1:5,:].detach().clone()
                self.text_prompts_class = x[:,5,:].detach().clone().unsqueeze(1)
                self.text_prompts_class = nn.Parameter(self.text_prompts_class)
                self.text_prompts_class_init = x[:,5,:].detach().clone().unsqueeze(1)
                self.text_prompts_end = x[:,6:,:].detach().clone()
            elif self.args.learnable_text == "S+a+cls+E":
                self.text_prompts_S = x[:,0,:].detach().clone().unsqueeze(1)
                self.text_prompts_S = nn.Parameter(self.text_prompts_S)
                self.text_prompts_S_init = x[:,0,:].detach().clone().unsqueeze(1)
                self.text_prompts_a = x[:,1:5,:].detach().clone()
                self.text_prompts_a = nn.Parameter(self.text_prompts_a)
                self.text_prompts_a_init = x[:,1:5,:].detach().clone()
                self.text_prompts_class = x[:,5,:].detach().clone().unsqueeze(1)
                self.text_prompts_class = nn.Parameter(self.text_prompts_class)
                self.text_prompts_class_init = x[:,5,:].detach().clone().unsqueeze(1)
                self.text_prompts_E = x[:,6,:].detach().clone().unsqueeze(1)
                self.text_prompts_E = nn.Parameter(self.text_prompts_E)
                self.text_prompts_E_init = x[:,6,:].detach().clone().unsqueeze(1)
                self.text_prompts_end = x[:,7:,:].detach().clone()
            elif self.args.learnable_text == "all":
                self.text_prompts = nn.Parameter(x)
                self.text_prompts_init = x.detach().clone()
        elif self.args.learn_all:
            # initialize multiple text prompts
            if self.args.learnable_text == "a":
                self.classnames = classnames
                # set as ctx_inits=a_photo_of_a_CLS,a_low_resolution_photo_of_a_CLS
                ctx_inits = self.args.ctx_init.split(',')
                n = len(ctx_inits) # the number of different prompts
                self.n_ensemble = n
                assert n > 1
                self.text_prompts_as = []
                self.text_prompts_a_inits = []
                self.text_prompts_begins = []
                self.text_prompts_ends = []
                self.text_prompts_classes = []
                self.text_prompts_tokens = []
                if len(self.xs) == 0:
                    self.xs = []
                    self.p_lens = []
                    for i in range(n):
                        ctx_inits[i] = ctx_inits[i].replace('_', ' ')
                        p_len = len(ctx_inits[i].split(' ')) - 1
                        x = [ctx_inits[i].replace("CLS", classname) for classname in classnames]
                        x = tokenize(x)
                        x = x.to("cuda")
                        text_prompts_token = x.detach().clone()
                        x = self.clip.token_embedding(x).type(self.dtype)  # [class_num, n_ctx, d_model]
                        self.xs.append(x)
                        self.p_lens.append(p_len)
                        self.text_prompts_tokens.append(text_prompts_token)
                    self.max_p_len = max(self.p_lens)
                for i in range(len(self.xs)):
                    x = self.xs[i]
                    # print(x.shape) # [200, 77, 512]
                    p_len = self.p_lens[i]
                    self.text_prompts_begin = x[:,0,:].detach().clone().unsqueeze(1) # ask about this
                    self.text_prompts_a = x[:,1:self.max_p_len+1,:].clone()
                    # self.text_prompts_a = nn.Parameter(self.text_prompts_a)
                    self.text_prompts_a_init = x[:,1:self.max_p_len+1,:].clone()
                    # print(self.text_prompts_a.shape, self.text_prompts_a_init.shape)
                    self.text_prompts_class = x[:,p_len+1,:].detach().clone().unsqueeze(1)
                    self.text_prompts_end = x[:,p_len+2:,:].detach().clone()
                    self.text_prompts_as.append(self.text_prompts_a.unsqueeze(0))
                    self.text_prompts_a_inits.append(self.text_prompts_a_init.unsqueeze(0))
                    self.text_prompts_begins.append(self.text_prompts_begin)
                    self.text_prompts_ends.append(self.text_prompts_end)
                    self.text_prompts_classes.append(self.text_prompts_class)
                # self.text_prompts_as = nn.ParameterList([nn.Parameter(t_a) for t_a in self.text_prompts_as])
                self.text_prompts_as = torch.cat(self.text_prompts_as, dim=0).detach()
                self.text_prompts_a_inits = torch.cat(self.text_prompts_a_inits, dim=0).detach()
                self.text_prompts_a = nn.Parameter(self.text_prompts_as)
        else:
            print("Initializing multiple text prompts, but only one is learnable")
            if self.args.learnable_text == "a":
                self.classnames = classnames
                # set as ctx_inits=a_photo_of_a_CLS,a_low_resolution_photo_of_a_CLS A
                ctx_inits = self.args.ctx_init.split(',')
                n = len(ctx_inits) # the number of different prompts
                self.n_ensemble = n
                assert n > 1
                self.text_prompts_as = []
                self.text_prompts_a_inits = []
                self.text_prompts_begins = []
                self.text_prompts_ends = []
                self.text_prompts_classes = []
                self.text_prompts_tokens = []
                text_ensemble_features = []
                if len(self.xs) == 0:
                    self.xs = []
                    self.p_lens = []
                    for i in range(n):
                        ctx_inits[i] = ctx_inits[i].replace('_', ' ')
                        p_len = len(ctx_inits[i].split(' ')) - 1
                        x = [ctx_inits[i].replace("CLS", classname) for classname in classnames]
                        x = tokenize(x)
                        x = x.to("cuda")
                        text_prompts_token = x.detach().clone()
                        if i == 0:
                            self.text_prompts_token = x.detach().clone()
                        x = self.clip.token_embedding(x).type(self.dtype)  # [class_num, n_ctx, d_model]
                        self.xs.append(x)
                        self.p_lens.append(p_len)
                        self.text_prompts_tokens.append(text_prompts_token)
                    self.max_p_len = max(self.p_lens)
                for i in range(len(self.xs)):
                    x = self.xs[i]
                    # print(x.shape) # [200, 77, 512]
                    p_len = self.p_lens[i]
                    if i == 0:
                        self.text_prompts_begin = x[:,0,:].detach().clone().unsqueeze(1)
                        self.text_prompts_a_ = x[:,1:p_len+1,:].clone()
                        self.text_prompts_a = nn.Parameter(self.text_prompts_a_)
                        self.text_prompts_a_init = x[:,1:p_len+1,:].clone()
                        self.text_prompts_class = x[:,p_len+1,:].detach().clone().unsqueeze(1)
                        self.text_prompts_end = x[:,p_len+2:,:].detach().clone()
                    else:
                        text_prompts_begin_ = x[:,0,:].detach().clone().unsqueeze(1)
                        text_prompts_a_ = x[:,1:p_len+1,:].clone()
                        text_prompts_class_ = x[:,p_len+1,:].detach().clone().unsqueeze(1)
                        text_prompts_end_ = x[:,p_len+2:,:].detach().clone()
                        text_prompts_token = self.text_prompts_tokens[i]
                        with torch.no_grad():
                            text_ensemble_features.append(self.get_text_features(text_prompts_a=text_prompts_a_, text_prompts_begin=text_prompts_begin_, text_prompts_end=text_prompts_end_, text_prompts_class=text_prompts_class_, text_prompts_token=text_prompts_token).unsqueeze(0))
                text_ensemble_features = torch.cat(text_ensemble_features, dim=0)
                self.text_ensemble_feature = text_ensemble_features.mean(dim=0)
                del text_ensemble_features
                # del x
                del self.xs
                del self.text_prompts_tokens
                # print('yes')
                # self.text_prompts_as = nn.ParameterList([nn.Parameter(t_a) for t_a in self.text_prompts_as])
                # self.text_prompts_a_inits = torch.cat(self.text_prompts_a_inits, dim=0).detach()


        del x
        return

    def similarity(self, q, k, topN=1):
        q = nn.functional.normalize(q, dim=-1)  # q shape [batch_size, 512]
        k = nn.functional.normalize(k, dim=-1)  # k shape [pool_size, 512]
        sim = torch.matmul(q, k.T)  # (B, T)
        # if self.args.prompt_penalty == 0 :
        if self.args.prompt_penalty == 0 or self.train_mode == False:
            dist = 1 - sim
        # elif self.args.prompt_penalty == 1 :
        elif self.args.prompt_penalty == 1 and self.train_mode == True:
            prompt_selected_sum = torch.Tensor(self.prompt_selected_sum_train)
            prompt_selected_sum = prompt_selected_sum.to('cuda')
            total = torch.sum(prompt_selected_sum)
            if total == 0:
                freq = prompt_selected_sum / 1
            else:
                freq = prompt_selected_sum / total
            dist = 1 - sim
            # dist = dist + freq * self.args.pool_size * 0.1
            dist = dist + freq * self.args.pool_size * 0.05
            # dist = dist + freq * self.args.pool_size * 0.5
            # dist = dist + freq * torch.exp(-total)
        val, idx = torch.topk(dist, topN, dim=1, largest=False)
        dist_pick = []
        for b in range(idx.shape[0]):
            pick = []
            for i in range(idx.shape[1]):
                pick.append(dist[b][idx[b][i]])
            dist_pick.append(torch.stack(pick))
        dist = torch.stack(dist_pick)
        # print("idx:", idx)
        return dist, idx

    def get_ensemble_text_features(self, dummy=None):
        n_ensemble = self.text_prompts_a.shape[0]
        assert n_ensemble == self.n_ensemble
        ensemble_text_features = []
        for i in range(n_ensemble):
            # print(self.text_prompts_a.shape)
            # print(self.p_lens)
            # text_prompts_a = self.text_prompts_a[i, :, :self.p_lens[i], :]
            text_prompts_begin = self.text_prompts_begins[i]
            text_prompts_end = self.text_prompts_ends[i]
            text_prompts_class = self.text_prompts_classes[i]
            text_prompts_token = self.text_prompts_tokens[i]
            ensemble_text_features.append(self.get_text_features(text_prompts_a=self.text_prompts_a[i, :, :self.p_lens[i], :], text_prompts_begin=text_prompts_begin, text_prompts_end=text_prompts_end, text_prompts_class=text_prompts_class, text_prompts_token=text_prompts_token).unsqueeze(0))
        ensemble_text_features = torch.cat(ensemble_text_features, dim=0)
        # print(ensemble_text_features.shape)
        ensemble_text_features = ensemble_text_features.mean(dim=0)
        # print(ensemble_text_features.shape, 'ensemble shape after')
        return ensemble_text_features

    def get_text_features(self, dummy=None, group_text_prompts=None, text_prompts_a=None, text_prompts_begin=None, text_prompts_end=None, text_prompts_class=None, text_prompts_token=None):
        # TODO
        # print("self.image_prompts.shape")
        # print(self.image_prompts.shape)
        # print("self.text_prompts_a.shape")
        # print(self.text_prompts_a.shape)
        # input()
        if text_prompts_a is None: # then ensemble is disabled
            text_prompts_a = self.text_prompts_a
            text_prompts_begin = self.text_prompts_begin
            text_prompts_end = self.text_prompts_end
            text_prompts_class = self.text_prompts_class
            text_prompts_token = self.text_prompts_token
        if self.args.learnable_text == "a" or self.args.learnable_text == "a+cls":
            # print(self.text_prompts_a.shape, 'text prompt')
            x = torch.cat(( text_prompts_begin, 
                            text_prompts_a,
                            text_prompts_class,
                            text_prompts_end), dim=1)     # [batch_size, n_ctx, d_model]
        class_num, _, dim = x.shape
        # print(x.shape)
        x = x + self.clip.positional_embedding.type(self.clip.dtype)
        # if self.args.share_prompts == 1:
        #     # print(self.share_prompts.shape)
        #     # input()
        #     x = torch.cat(( x[:,:6,:], 
        #                     self.share_prompts.repeat(class_num, 1, 1)[:,:,:512],
        #                     x[:,6,:].unsqueeze(1),
        #                     x[:,8:,:]), dim=1)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # print(x.shape)  # [n_ctx, class_num, d_model]
        x = self.clip.transformer(x)
        # print(x.shape)  # [n_ctx, class_num, d_model]
        x = x.permute(1, 0, 2)  # LND -> NLD
        # print(x.shape)  # [class_num, n_ctx, d_model]
        x = self.clip.ln_final(x).type(self.clip.dtype)
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text_prompts_token.argmax(dim=-1)] @ self.clip.text_projection


        return x

    
    def get_image_embedding(self, x):
        x = self.image_encoder.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        # shape = [*, grid ** 2 + 1, width]
        # print(x.shape)
        # print(self.image_encoder.class_embedding.shape)
        x = torch.cat([self.image_encoder.class_embedding.to(x.dtype) + \
            torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  
        # print(x.shape)
        # input()
        x = x + self.image_encoder.positional_embedding.to(x.dtype)
        return x

    def get_image_features(self, x, prompts=None):
        if prompts is not None:
            x = torch.cat((x[:,0,:].unsqueeze(1), prompts, x[:,1:,:]), dim=1)  # [64, 197+topN*plen, 768]
        # x shape [64, 196(197), 768]
        x = self.image_encoder.ln_pre(x)            # x shape [64, 196(197), 768]
        x = x.permute(1, 0, 2)  # NLD -> LND        # x shape [196(197), 64, 768]
        x = self.image_encoder.transformer(x)       # x shape [196(197), 64, 768]
        x = x.permute(1, 0, 2)  # LND -> NLD        # x shape [64, 196(197), 768]
        x = self.image_encoder.ln_post(x[:, 0, :])  # x shape [64, 768]
        if self.image_encoder.proj is not None:
            x = x @ self.image_encoder.proj         # x shape [64, 512]
        return x
        

    ### use for prefix tuning 
    def get_image_features_prefix(self, x, prompts=None):
        x = self.image_encoder.ln_pre(x)            # x shape [64, 196(197), 768]
        x = x.permute(1, 0, 2)  # NLD -> LND        # x shape [196(197), 64, 768]
        # print('get_image_features_prefix') # the same
        # print(x)
        x, dist = self.transformer_forward_prefix(x, prompts)       # x shape [196(197), 64, 768]
        # x = self.clip.visual.transformer(x)       # x shape [196(197), 64, 768]
        # print('get_image_features_prefix') # not the same!
        # print(x)
        x = x.permute(1, 0, 2)  # LND -> NLD        # x shape [64, 196(197), 768]
        x = self.image_encoder.ln_post(x[:, 0, :])  # x shape [64, 768]
        # print('get_image_features_prefix') # not the same
        # print(x)
        if self.image_encoder.proj is not None:
            x = x @ self.image_encoder.proj         # x shape [64, 512]
        return x, dist
    
    ### use for prefix tuning
    def transformer_forward_prefix(self, x, prompts=None):
        Transformer = self.clip.visual.transformer
        dist = 0
        self.att_map = None
        # print('reset att map')
        for layer in range(Transformer.layers):
            if layer in self.image_prompt_layers:
                idx = self.image_prompt_layers.index(layer)
                prompts_one_layer = prompts[:, idx, :, :] #[batch, , plen, dim]
            else:
                prompts_one_layer = None
            # print('layer', layer)
            x = self.ResidualAttentionBlock_forward_prefix(x, layer, prompts_one_layer)
            # print('transformer_forward_prefix') # not the same
            # if layer <= 2:
            #     print(layer)
            #     print(x[0, 0, :100])
        return x, dist
    
    def get_image_features_prefix_layer(self, x, prompts=None, layers=[10, 11]):
        x = self.image_encoder.ln_pre(x)            # x shape [64, 196(197), 768]
        x = x.permute(1, 0, 2)  # NLD -> LND        # x shape [196(197), 64, 768]
        # print('get_image_features_prefix') # the same
        # print(x)
        outputs, dist = self.transformer_forward_prefix_layer(x, prompts, layers)       # x shape [196(197), 64, 768]
        
        # x = self.clip.visual.transformer(x)       # x shape [196(197), 64, 768]
        # print('get_image_features_prefix') # not the same!
        # print(x)
        layer_features = []
        patch_features = outputs[-1][1:, :, :]
        class_token_feature = outputs[-1][0, :, :].unsqueeze(0)
        for x1 in outputs:
            # print(x1.shape, 'x1 shape') # [197, 64, 768]
            x = x1.permute(1, 0, 2)  # LND -> NLD        # x shape [64, 196(197), 768]
            x = self.image_encoder.ln_post(x[:, 0, :])  # x shape [64, 768]
            # print('get_image_features_prefix') # not the same
            # print(x)
            if self.image_encoder.proj is not None:
                x = x @ self.image_encoder.proj         # x shape [64, 512]
            layer_features.append(x.unsqueeze(1))
        layer_features = torch.cat(layer_features, dim=1)
        # print(layer_features.shape, 'layer_features') # shape [64, 2, 512]
        return layer_features, dist, patch_features, class_token_feature
    
    ### use for prefix tuning
    def transformer_forward_prefix_layer(self, x, prompts=None, layers=[10, 11]): # layer num total 12
        Transformer = self.clip.visual.transformer
        dist = 0
        
        print('layer!!!')
        outputs = []
        for layer in range(Transformer.layers):
            if layer in self.image_prompt_layers:
                idx = self.image_prompt_layers.index(layer)
                prompts_one_layer = prompts[:, idx, :, :] #[batch, , plen, dim]
            else:
                prompts_one_layer = None
            x = self.ResidualAttentionBlock_forward_prefix(x, layer, prompts_one_layer)
            if layer in layers:
                outputs.append(x)
            # print('transformer_forward_prefix') # not the same
            # if layer <= 2:
            #     print(layer)
            #     print(x[0, 0, :100])
        return outputs, dist


    ### use for prefix tuning
    def ResidualAttentionBlock_forward_prefix(self, x, layer, prompts=None):
        ### Block: ResidualAttentionBlock self
        Block = self.clip.visual.transformer.resblocks[layer]
        # print('pormpts in ResidualAttentionBlock_forward_prefix')
        # print(prompts) # the same
        # print(Block)
        # input()
        q = Block.ln_1(x)
        Block.attn_mask = Block.attn_mask.to(dtype=q.dtype, device=q.device) if Block.attn_mask is not None else None
        if prompts is None:
            x = x + Block.attn(q, q, q, need_weights=False, attn_mask=Block.attn_mask)[0] #original
            # x = Block.attn(q, q, q, need_weights=False, attn_mask=Block.attn_mask)
            # # print(len(x1), 'len x1') # 2
            # print(len(x1))
            # x = x + x1[0]
            # # a = x1[1].permute(1, 0, 2)[0, :, 1:].unsqueeze(0)

            q1 = q.permute(1, 0, 2)[:, 1:, :]
            v1 = q.permute(1, 0, 2) # [support set, 197, 768]
            v1 = v1[:, 0, :].unsqueeze(1) # [support set, 1, 768]
            clique_size = q1.shape[0]
            q1 = q1.permute(0, 2, 1)
            a = torch.matmul(v1, q1)
            a = a.squeeze().unsqueeze(0)
            if clique_size == 1:
                a = a.unsqueeze(0)
            # print(a.shape) 
            # print('here')
            if self.args.att_map and self.batch_learning and q.shape[1] >= 1:
                with torch.no_grad():
                    if self.att_map is None:
                        self.att_map = a.clone() # [1, support set, 196]
                    else:
                        # print(self.att_map.shape, a.shape)
                        self.att_map = torch.cat([self.att_map, a], dim=0)
            # print('ResidualAttentionBlock_forward_prefix') # not the same
            # if layer >= 10: print
            #     print(x)
            # attention_return = Block.attn(x, x, x, need_weights=False, attn_mask=Block.attn_mask)[0]
        else:
            prompts = prompts.permute(1, 0, 2)
            # print(prompts.shape)
            # input()
            half = int(prompts.shape[0]/2)
            # print(prompts[:half, :, :].unsqueeze(1).shape)
            # print(prompts[:half, :, :].shape)
            # print(prompts[:half, :, :].unsqueeze(1).shape)
            # print(prompts[:half, :, :].shape)
            # print(q[0,:,:].unsqueeze(0).shape)
            # print(q[1:,:,:].shape)
            # input()
            k = torch.cat([q[0,:,:].unsqueeze(0), prompts[:half, :, :] , q[1:,:,:]], 0)
            v = torch.cat([q[0,:,:].unsqueeze(0), prompts[half:, :, :] , q[1:,:,:]], 0)
            # print(k.requires_grad, v.requires_grad)
            # k = torch.cat([x[:,0,:].unsqueeze(1), prompts[:, :half ,:] , x[:,1:,:]], 1)
            # v = torch.cat([x[:,0,:].unsqueeze(1), prompts[:, half: ,:] , x[:,1:,:]], 1)
            # print('attention! k ,q, v: ', k.shape, q.shape, v.shape)
            # for attention map visualization:
            # if self.args.att_map and self.batch_learning and k.shape[1] >= 1:
            #     with torch.no_grad():
            #         q1 = q.permute(1, 0, 2)
            #         v1 = v.permute(1, 0, 2) # [support set, 197, 768]
            #         q1 = q1[:, 0, :].unsqueeze(1) # [support set, 1, 768]
            #         q1 = q1.permute(0, 2, 1)
            #         a = torch.matmul(v1, q1).squeeze()[:, :196].unsqueeze(0)
            #         if self.att_map is None:
            #             print('is none')
            #             self.att_map = a # [1, support set, 196]
            #         else:
            #             # print(self.att_map.shape, a.shape)
            #             self.att_map = torch.cat([self.att_map, a], dim=0)
            

            x = x + Block.attn(q, k, v, need_weights=False, attn_mask=Block.attn_mask)[0]
            # print('ResidualAttentionBlock_forward_prefix')
            # print(x)
        # x = x + attention_return
        x = x + Block.mlp(Block.ln_2(x))
        return x


    def cluster_features(self, x):
        return self.get_image_features(self.get_image_embedding(x))

    def inference(self, x, group_image_prompts=None, group_text_prompts=None):
        self.att_map = None
        if self.args.simam:
            x = self.simam(x)
        batch = x.shape[0]
        dist = 0
        with torch.no_grad():
            x = self.get_image_embedding(x) # embedding shape: [batch_size, 197, 768]
            # print('image_embedding')
            # print(x)
        # image_prompts, text_prompts, dist = self.get_prompts(embedding[:,1:,:])
        ### if use image prompts
        # print('self image prompt shape: ', self.image_prompts.shape) # [1, 1, 2, 768]
        # print('self text  prompt shape: ', self.text_prompts_a.shape) # [200, 6, 512]
        if self.args.image_prompts:
            # print(self.image_prompts.repeat(24,1,1).shape)
            # input()
            ### if use Prefix Tuning
            if self.args.prefix_tuning:
                # here for A
                # if self.args.test_sets in ['V', 'K', 'C', 'A']:
                if self.use_sum:
                    if self.args.test_sets in ['V', 'C', 'K', 'A']:
                        x1, dist = checkpoint(self.get_image_features_prefix, x, self.image_prompts.repeat(batch,1,1,1))
                        # x, dist = self.get_image_features_prefix(x, self.image_prompts.repeat(batch,1,1,1))
                    else:
                        # print('image_prompts')
                        # print(self.image_prompts.repeat(batch,1,1,1))
                        x1, dist = self.get_image_features_prefix(x, self.image_prompts.repeat(batch,1,1,1))
                else:
                    if group_image_prompts is not None:
                        final_image_prompts = torch.cat([self.image_prompts, group_image_prompts], dim=2)
                    else:
                        final_image_prompts = self.image_prompts
                    if self.args.parallel:
                        x1, dist = self.get_image_features_prefix(x, final_image_prompts.repeat(batch,1,1,1))
                    else:
                        x1, dist = checkpoint(self.get_image_features_prefix, x, final_image_prompts.repeat(batch,1,1,1))
                        # x1, dist = self.get_image_features_prefix(x, final_image_prompts.repeat(batch,1,1,1)) 
            else:    
                # TODO
                if batch == 1:
                    x1 = self.get_image_features(x, self.image_prompts[0].unsqueeze(0))
                else:
                    x1 = self.get_image_features(x, self.image_prompts)
            # print('image_embedding') #not the same
            # print(x)
        else:
            # x = checkpoint(self.get_image_features, x)
            # with torch.no_grad():
            #     x = self.get_image_features(x)
            x1 = self.get_image_features(x)

        x = x1 / x1.norm(dim=-1, keepdim=True)
        # logit_scale = self.logit_scale.exp()
        # logits per image
        # logits = logit_scale * x @ x.t()
        # return logits, dist
        # return logits, x, x
        # if self.args.test_sets in ['V', 'K', 'C', 'A']:
        # if self.args.test_sets in ['V', 'K', 'C', 'A']:
        if not self.args.ensemble:
            text_features = checkpoint(self.get_text_features, self.dummy, group_text_prompts)
            # text_features = self.get_text_features(self.dummy, group_text_prompts)
        elif self.args.learn_all:
            text_features = checkpoint(self.get_ensemble_text_features, self.dummy)
        else:
            if self.args.parallel:
                text_features = self.get_text_features(self.dummy)
            else:
                text_features = checkpoint(self.get_text_features, self.dummy)
                # text_features = self.get_text_features(self.dummy)
            # print(text_features.shape, "text_features") #[200,512]
            # print(self.text_ensemble_feature.shape, "ensemble features") #[200,512]
            # text_features1 = text_features / text_features.norm(dim=1, keepdim=True)
            # self.text_ensemble_feature1 = self.text_ensemble_feature / self.text_ensemble_feature.norm(dim=1, keepdim=True)
            # print(text_features1.T @ self.text_ensemble_feature1)
            if self.args.use_ensemble:
                text_features = (1 - self.args.ensemble_rate) * text_features + self.args.ensemble_rate * self.text_ensemble_feature.clone()

        # else:
        #     text_features = self.get_text_features()
        # text_features = self.get_text_features()
         
        # print(text_features.shape)
        # input()
        # print(text_features.norm(dim=1, keepdim=True))
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        # logits per image
        logits = logit_scale * x @ text_features.t()
        # return logits, dist
        return logits, x, text_features
    
    def inference_layer(self, x, group_image_prompts=None, layers=[10, 11]):
        batch = x.shape[0]
        dist = 0
        with torch.no_grad():
            x = self.get_image_embedding(x) # embedding shape: [batch_size, 197, 768]
            # print('image_embedding')
            # print(x)
        # image_prompts, text_prompts, dist = self.get_prompts(embedding[:,1:,:])
        ### if use image prompts
        if self.args.image_prompts:
            # print(self.image_prompts.repeat(24,1,1).shape)
            # input()
            ### if use Prefix Tuning
            if self.args.prefix_tuning:
                # here for A
                # if self.args.test_sets in ['V', 'K', 'C', 'A']:
                if self.use_sum:
                    x, dist, patch_features, class_token_feature = checkpoint(self.get_image_features_prefix_layer, x, self.image_prompts.repeat(batch,1,1,1), layers)
                    # x, dist = self.get_image_features_prefix(x, self.image_prompts.repeat(batch,1,1,1))
                else:
                    if group_image_prompts is not None:
                        final_image_prompts = torch.cat([self.image_prompts, group_image_prompts], dim=2)
                    else:
                        final_image_prompts = self.image_prompts
                    x, dist, patch_features, class_token_feature = checkpoint(self.get_image_features_prefix_layer, x, final_image_prompts.repeat(batch,1,1,1), layers)
            else:    
                # TODO
                if batch == 1:
                    x = self.get_image_features(x, self.image_prompts[0].unsqueeze(0))
                else:
                    x = self.get_image_features(x, self.image_prompts)
            # print('image_embedding') #not the same
            # print(x)
        else:
            # x = checkpoint(self.get_image_features, x)
            # with torch.no_grad():
            #     x = self.get_image_features(x)
            x = self.get_image_features(x)

        x = x / x.norm(dim=-1, keepdim=True)
        x_last = x[:, -1, :]
        print('x', x.shape)
        # if self.args.test_sets in ['V', 'K', 'C', 'A']:
        if self.args.test_sets in ['V', 'K', 'C', 'A']:
            text_features = checkpoint(self.get_text_features, self.dummy)
        else:
            text_features = self.get_text_features()
        # text_features = self.get_text_features()
         
        # print(text_features.shape)
        # input()
        # print(text_features.norm(dim=1, keepdim=True))
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        # logits per image
        logits = logit_scale * x_last @ text_features.t()
        # return logits, dist
        return logits, x, text_features, patch_features, class_token_feature
     
    def forward_clip(self, image):
        with torch.no_grad():
            image_features = self.clip.encode_image(image)
            self.tokenized_text = self.tokenized_text.to('cuda')
            # print(self.tokenized_text.device)
            text_features = self.clip.encode_text(self.tokenized_text)

            # normalized features
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)

            # cosine similarity as logits
            logit_scale = self.clip.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.t()
            # logits_per_text = logits_per_image.t()

            # shape = [global_batch_size, global_batch_size]
            return logits_per_image

    def forward(self, input, group_image_prompts=None, group_text_prompts=None, train_mode=True):
        return self.inference(input, group_image_prompts=group_image_prompts, group_text_prompts=group_text_prompts)
    






class BlipTestTimeTuning(nn.Module):
    def __init__(
        self, args, device, classnames,
        criterion='cosine', arch="ViT-L/14"
    ):
        super(BlipTestTimeTuning, self).__init__()
        self.blip_model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        clip, _, preprocess = load(arch, device=device, download_root=DOWNLOAD_ROOT)
        self.preprocess = preprocess
        self.clip = clip
        self.image_encoder = self.blip_model.vision_model
        # self.text_encoder = TextEncoder(clip)
        self.logit_scale = clip.logit_scale.data
        self.criterion = criterion
        self.args = args
        self.train_mode = True
        self.xs = []
        self.p_lens = []
        self.att_map = None
        self.simam = SimamModule()
        self.batch_learning = False
        self.dummy = torch.ones(1, dtype=torch.float32, requires_grad=True)
        if self.args.image_prompts:
            self.image_prompt_layers = self.args.image_prompt_layer
            if self.args.prefix_tuning:
                if self.args.arch == "ViT-L/14":
                    self.image_prompts = torch.empty(
                    (1, len(self.image_prompt_layers), 2, 1024), 
                    dtype=self.clip.dtype,
                    device="cuda").uniform_(-1, 1).requires_grad_(True)
                    self.use_sum = args.use_sum
                else:
                    
                    self.image_prompts = torch.empty(
                        # TODO
                        ### per image uses the same prompt
                        (1, len(self.image_prompt_layers), 2, 768), 
                        ### per image uses different prompt
                        # (self.args.batch_size, len(self.image_prompt_layers), 2, 768), 
                        dtype=self.clip.dtype,
                        device="cuda").uniform_(-1, 1).requires_grad_(True)
                    self.use_sum = args.use_sum
            else:
                self.image_prompts = torch.empty(
                    # TODO
                    # (1, 1, 768), 
                    (self.args.batch_size, 1, 768), 
                    dtype=self.clip.dtype,
                    device="cuda").uniform_(-1, 1).requires_grad_(True)
            self.image_prompts = nn.Parameter(self.image_prompts)
        
    @property
    def dtype(self):
        return self.image_encoder.conv1.weight.dtype
    
    def reset(self):
        self.reset_Tclass_prompts()
        # print(model.image_prompts.requires_grad)
        if self.args.image_prompts and self.args.reset_image_prompts is True:
            # print(model.image_prompts.requires_grad)
            self.reset_image_prompts()
            # print(model.image_prompts.requires_grad)

    def reset_share_prompts(self):
        self.share_prompts.uniform_(-1, 1).requires_grad_(True)
    
    def reset_image_prompts(self):
        # print("reset image prompts")
        # print(self.image_prompts[0:10])
        self.image_prompts.uniform_(-1, 1).requires_grad_(True)
        
    def reset_Tclass_prompts(self):
        # print("reset Tclass prompts")
        # TODO
        if self.args.CSTP == 1:
            if self.args.learnable_text == "a":
                if self.args.ensemble and self.args.learn_all:
                    self.text_prompts_a.copy_(self.text_prompts_a_inits)
                    # for i, text_prompts_a in enumerate(self.text_prompts_as):
                    #     text_prompts_a.copy_(self.text_prompts_a_inits[i])
                else:
                    self.text_prompts_a.copy_(self.text_prompts_a_init)
            elif self.args.learnable_text == "a+cls":
                self.text_prompts_a.copy_(self.text_prompts_a_init)
                self.text_prompts_class.copy_(self.text_prompts_class_init)
            elif self.args.learnable_text == "S+a+cls+E":
                self.text_prompts_S.copy_(self.text_prompts_S_init)
                self.text_prompts_a.copy_(self.text_prompts_a_init)
                self.text_prompts_class.copy_(self.text_prompts_class_init)
                self.text_prompts_E.copy_(self.text_prompts_E_init)
            elif self.args.learnable_text == "all":
                self.text_prompts.copy_(self.text_prompts_init)
        elif self.args.CSTP == 2:
            if self.args.learnable_text == "a":
                self.CSTP_bvector.copy_(self.text_prompts_a_init)
                text_prompts_a = torch.matmul(self.CSTP_weight, self.CSTP_bvector.reshape(self.args.CSTP_N, -1))
                text_prompts_a = text_prompts_a.reshape(len(self.classnames), 4, -1)
                self.text_prompts_a.copy_(text_prompts_a)
        return



    def init_text_prompts(self, classnames):        
        # 如果每个类使用一个单独的文本prompt
        if not self.args.ensemble:
            self.classnames = classnames
            # print(self.args.ctx_init)
            # print(self.args.n_ctx)
            ctx_init = self.args.ctx_init.replace("_", " ")
            ctx_init = self.args.ctx_init.replace("-", " ")
            # print(ctx_init)
            # input()
            # x = ['This is a photo of a '+classname for classname in classnames] 
            # p_len = 6
            x = [ctx_init.replace("CLS", classname) for classname in classnames] 
            # print(x) 
            # input()
            # x = ['a photo of a'+classname for classname in classnames] 
            p_len = self.args.n_ctx
            # x = ["X" * self.args.text_plen] + x
            # x = ["X " + p for p in x]
            # x = ["X " * self.args.text_plen + p for p in x]
            x = tokenize(x) # [class_num, n_ctx]
            self.tokenized_text = x.detach().clone()
            x = x.to("cuda")
            self.text_prompts_token = x.detach().clone()
            x = self.clip.token_embedding(x).type(self.clip.dtype)  # [class_num, n_ctx, d_model]
            # TODO
            if self.args.learnable_text == "a":
                self.text_prompts_begin = x[:,0,:].detach().clone().unsqueeze(1)
                self.text_prompts_a = x[:,1:p_len+1,:].detach().clone()
                self.text_prompts_a = nn.Parameter(self.text_prompts_a)
                self.text_prompts_a_init = x[:,1:p_len+1,:].detach().clone()
                self.text_prompts_class = x[:,p_len+1,:].detach().clone().unsqueeze(1)
                self.text_prompts_end = x[:,p_len+2:,:].detach().clone()
            elif self.args.learnable_text == "a+cls":
                self.text_prompts_begin = x[:,0,:].detach().clone().unsqueeze(1)
                self.text_prompts_a = x[:,1:5,:].detach().clone()
                self.text_prompts_a = nn.Parameter(self.text_prompts_a)
                self.text_prompts_a_init = x[:,1:5,:].detach().clone()
                self.text_prompts_class = x[:,5,:].detach().clone().unsqueeze(1)
                self.text_prompts_class = nn.Parameter(self.text_prompts_class)
                self.text_prompts_class_init = x[:,5,:].detach().clone().unsqueeze(1)
                self.text_prompts_end = x[:,6:,:].detach().clone()
            elif self.args.learnable_text == "S+a+cls+E":
                self.text_prompts_S = x[:,0,:].detach().clone().unsqueeze(1)
                self.text_prompts_S = nn.Parameter(self.text_prompts_S)
                self.text_prompts_S_init = x[:,0,:].detach().clone().unsqueeze(1)
                self.text_prompts_a = x[:,1:5,:].detach().clone()
                self.text_prompts_a = nn.Parameter(self.text_prompts_a)
                self.text_prompts_a_init = x[:,1:5,:].detach().clone()
                self.text_prompts_class = x[:,5,:].detach().clone().unsqueeze(1)
                self.text_prompts_class = nn.Parameter(self.text_prompts_class)
                self.text_prompts_class_init = x[:,5,:].detach().clone().unsqueeze(1)
                self.text_prompts_E = x[:,6,:].detach().clone().unsqueeze(1)
                self.text_prompts_E = nn.Parameter(self.text_prompts_E)
                self.text_prompts_E_init = x[:,6,:].detach().clone().unsqueeze(1)
                self.text_prompts_end = x[:,7:,:].detach().clone()
            elif self.args.learnable_text == "all":
                self.text_prompts = nn.Parameter(x)
                self.text_prompts_init = x.detach().clone()
        elif self.args.learn_all:
            # initialize multiple text prompts
            if self.args.learnable_text == "a":
                self.classnames = classnames
                # set as ctx_inits=a_photo_of_a_CLS,a_low_resolution_photo_of_a_CLS
                ctx_inits = self.args.ctx_init.split(',')
                n = len(ctx_inits) # the number of different prompts
                self.n_ensemble = n
                assert n > 1
                self.text_prompts_as = []
                self.text_prompts_a_inits = []
                self.text_prompts_begins = []
                self.text_prompts_ends = []
                self.text_prompts_classes = []
                self.text_prompts_tokens = []
                if len(self.xs) == 0:
                    self.xs = []
                    self.p_lens = []
                    for i in range(n):
                        ctx_inits[i] = ctx_inits[i].replace('_', ' ')
                        p_len = len(ctx_inits[i].split(' ')) - 1
                        x = [ctx_inits[i].replace("CLS", classname) for classname in classnames]
                        x = tokenize(x)
                        x = x.to("cuda")
                        text_prompts_token = x.detach().clone()
                        x = self.clip.token_embedding(x).type(self.dtype)  # [class_num, n_ctx, d_model]
                        self.xs.append(x)
                        self.p_lens.append(p_len)
                        self.text_prompts_tokens.append(text_prompts_token)
                    self.max_p_len = max(self.p_lens)
                for i in range(len(self.xs)):
                    x = self.xs[i]
                    # print(x.shape) # [200, 77, 512]
                    p_len = self.p_lens[i]
                    self.text_prompts_begin = x[:,0,:].detach().clone().unsqueeze(1) # ask about this
                    self.text_prompts_a = x[:,1:self.max_p_len+1,:].clone()
                    # self.text_prompts_a = nn.Parameter(self.text_prompts_a)
                    self.text_prompts_a_init = x[:,1:self.max_p_len+1,:].clone()
                    # print(self.text_prompts_a.shape, self.text_prompts_a_init.shape)
                    self.text_prompts_class = x[:,p_len+1,:].detach().clone().unsqueeze(1)
                    self.text_prompts_end = x[:,p_len+2:,:].detach().clone()
                    self.text_prompts_as.append(self.text_prompts_a.unsqueeze(0))
                    self.text_prompts_a_inits.append(self.text_prompts_a_init.unsqueeze(0))
                    self.text_prompts_begins.append(self.text_prompts_begin)
                    self.text_prompts_ends.append(self.text_prompts_end)
                    self.text_prompts_classes.append(self.text_prompts_class)
                # self.text_prompts_as = nn.ParameterList([nn.Parameter(t_a) for t_a in self.text_prompts_as])
                self.text_prompts_as = torch.cat(self.text_prompts_as, dim=0).detach()
                self.text_prompts_a_inits = torch.cat(self.text_prompts_a_inits, dim=0).detach()
                self.text_prompts_a = nn.Parameter(self.text_prompts_as)
        else:
            print("Initializing multiple text prompts, but only one is learnable")
            if self.args.learnable_text == "a":
                self.classnames = classnames
                # set as ctx_inits=a_photo_of_a_CLS,a_low_resolution_photo_of_a_CLS A
                ctx_inits = self.args.ctx_init.split(',')
                n = len(ctx_inits) # the number of different prompts
                self.n_ensemble = n
                assert n > 1
                self.text_prompts_as = []
                self.text_prompts_a_inits = []
                self.text_prompts_begins = []
                self.text_prompts_ends = []
                self.text_prompts_classes = []
                self.text_prompts_tokens = []
                text_ensemble_features = []
                if len(self.xs) == 0:
                    self.xs = []
                    self.p_lens = []
                    for i in range(n):
                        ctx_inits[i] = ctx_inits[i].replace('_', ' ')
                        p_len = len(ctx_inits[i].split(' ')) - 1
                        x = [ctx_inits[i].replace("CLS", classname) for classname in classnames]
                        x = tokenize(x)
                        x = x.to("cuda")
                        text_prompts_token = x.detach().clone()
                        if i == 0:
                            self.text_prompts_token = x.detach().clone()
                        x = self.clip.token_embedding(x).type(self.dtype)  # [class_num, n_ctx, d_model]
                        self.xs.append(x)
                        self.p_lens.append(p_len)
                        self.text_prompts_tokens.append(text_prompts_token)
                    self.max_p_len = max(self.p_lens)
                for i in range(len(self.xs)):
                    x = self.xs[i]
                    # print(x.shape) # [200, 77, 512]
                    p_len = self.p_lens[i]
                    if i == 0:
                        self.text_prompts_begin = x[:,0,:].detach().clone().unsqueeze(1)
                        self.text_prompts_a_ = x[:,1:p_len+1,:].clone()
                        self.text_prompts_a = nn.Parameter(self.text_prompts_a_)
                        self.text_prompts_a_init = x[:,1:p_len+1,:].clone()
                        self.text_prompts_class = x[:,p_len+1,:].detach().clone().unsqueeze(1)
                        self.text_prompts_end = x[:,p_len+2:,:].detach().clone()
                    else:
                        text_prompts_begin_ = x[:,0,:].detach().clone().unsqueeze(1)
                        text_prompts_a_ = x[:,1:p_len+1,:].clone()
                        text_prompts_class_ = x[:,p_len+1,:].detach().clone().unsqueeze(1)
                        text_prompts_end_ = x[:,p_len+2:,:].detach().clone()
                        text_prompts_token = self.text_prompts_tokens[i]
                        with torch.no_grad():
                            text_ensemble_features.append(self.get_text_features(text_prompts_a=text_prompts_a_, text_prompts_begin=text_prompts_begin_, text_prompts_end=text_prompts_end_, text_prompts_class=text_prompts_class_, text_prompts_token=text_prompts_token).unsqueeze(0))
                text_ensemble_features = torch.cat(text_ensemble_features, dim=0)
                self.text_ensemble_feature = text_ensemble_features.mean(dim=0)
                del text_ensemble_features
                # del x
                del self.xs
                del self.text_prompts_tokens
                # print('yes')
                # self.text_prompts_as = nn.ParameterList([nn.Parameter(t_a) for t_a in self.text_prompts_as])
                # self.text_prompts_a_inits = torch.cat(self.text_prompts_a_inits, dim=0).detach()


        del x
        return

    def similarity(self, q, k, topN=1):
        q = nn.functional.normalize(q, dim=-1)  # q shape [batch_size, 512]
        k = nn.functional.normalize(k, dim=-1)  # k shape [pool_size, 512]
        sim = torch.matmul(q, k.T)  # (B, T)
        # if self.args.prompt_penalty == 0 :
        if self.args.prompt_penalty == 0 or self.train_mode == False:
            dist = 1 - sim
        # elif self.args.prompt_penalty == 1 :
        elif self.args.prompt_penalty == 1 and self.train_mode == True:
            prompt_selected_sum = torch.Tensor(self.prompt_selected_sum_train)
            prompt_selected_sum = prompt_selected_sum.to('cuda')
            total = torch.sum(prompt_selected_sum)
            if total == 0:
                freq = prompt_selected_sum / 1
            else:
                freq = prompt_selected_sum / total
            dist = 1 - sim
            # dist = dist + freq * self.args.pool_size * 0.1
            dist = dist + freq * self.args.pool_size * 0.05
            # dist = dist + freq * self.args.pool_size * 0.5
            # dist = dist + freq * torch.exp(-total)
        val, idx = torch.topk(dist, topN, dim=1, largest=False)
        dist_pick = []
        for b in range(idx.shape[0]):
            pick = []
            for i in range(idx.shape[1]):
                pick.append(dist[b][idx[b][i]])
            dist_pick.append(torch.stack(pick))
        dist = torch.stack(dist_pick)
        # print("idx:", idx)
        return dist, idx

    def get_ensemble_text_features(self, dummy=None):
        n_ensemble = self.text_prompts_a.shape[0]
        assert n_ensemble == self.n_ensemble
        ensemble_text_features = []
        for i in range(n_ensemble):
            # print(self.text_prompts_a.shape)
            # print(self.p_lens)
            # text_prompts_a = self.text_prompts_a[i, :, :self.p_lens[i], :]
            text_prompts_begin = self.text_prompts_begins[i]
            text_prompts_end = self.text_prompts_ends[i]
            text_prompts_class = self.text_prompts_classes[i]
            text_prompts_token = self.text_prompts_tokens[i]
            ensemble_text_features.append(self.get_text_features(text_prompts_a=self.text_prompts_a[i, :, :self.p_lens[i], :], text_prompts_begin=text_prompts_begin, text_prompts_end=text_prompts_end, text_prompts_class=text_prompts_class, text_prompts_token=text_prompts_token).unsqueeze(0))
        ensemble_text_features = torch.cat(ensemble_text_features, dim=0)
        # print(ensemble_text_features.shape)
        ensemble_text_features = ensemble_text_features.mean(dim=0)
        # print(ensemble_text_features.shape, 'ensemble shape after')
        return ensemble_text_features

    def get_text_features(self, dummy=None, group_text_prompts=None, text_prompts_a=None, text_prompts_begin=None, text_prompts_end=None, text_prompts_class=None, text_prompts_token=None):
        # TODO
        # print("self.image_prompts.shape")
        # print(self.image_prompts.shape)
        # print("self.text_prompts_a.shape")
        # print(self.text_prompts_a.shape)
        # input()
        if text_prompts_a is None: # then ensemble is disabled
            text_prompts_a = self.text_prompts_a
            text_prompts_begin = self.text_prompts_begin
            text_prompts_end = self.text_prompts_end
            text_prompts_class = self.text_prompts_class
            text_prompts_token = self.text_prompts_token
        if self.args.learnable_text == "a" or self.args.learnable_text == "a+cls":
            # print(self.text_prompts_a.shape, 'text prompt')
            x = torch.cat(( text_prompts_begin, 
                            text_prompts_a,
                            text_prompts_class,
                            text_prompts_end), dim=1)     # [batch_size, n_ctx, d_model]
        print(x.shape, "real")
        print(x)
        x = self.blip_model.get_text_features(x)
        # class_num, _, dim = x.shape
        # # print(x.shape)
        # x = x + self.clip.positional_embedding.type(self.clip.dtype)
        # # if self.args.share_prompts == 1:
        # #     # print(self.share_prompts.shape)
        # #     # input()
        # #     x = torch.cat(( x[:,:6,:], 
        # #                     self.share_prompts.repeat(class_num, 1, 1)[:,:,:512],
        # #                     x[:,6,:].unsqueeze(1),
        # #                     x[:,8:,:]), dim=1)
        # x = x.permute(1, 0, 2)  # NLD -> LND
        # # print(x.shape)  # [n_ctx, class_num, d_model]
        # x = self.clip.transformer(x)
        # # print(x.shape)  # [n_ctx, class_num, d_model]
        # x = x.permute(1, 0, 2)  # LND -> NLD
        # # print(x.shape)  # [class_num, n_ctx, d_model]
        # x = self.clip.ln_final(x).type(self.clip.dtype)
        # # take features from the eot embedding (eot_token is the highest number in each sequence)
        # x = x[torch.arange(x.shape[0]), text_prompts_token.argmax(dim=-1)] @ self.clip.text_projection


        return x

    
    def get_image_embedding(self, x):
        x = self.image_encoder.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        # shape = [*, grid ** 2 + 1, width]
        # print(x.shape)
        # print(self.image_encoder.class_embedding.shape)
        x = torch.cat([self.image_encoder.class_embedding.to(x.dtype) + \
            torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  
        # print(x.shape)
        # input()
        x = x + self.image_encoder.positional_embedding.to(x.dtype)
        return x

    def get_image_features(self, x, prompts=None):
        if prompts is not None:
            x = torch.cat((x[:,0,:].unsqueeze(1), prompts, x[:,1:,:]), dim=1)  # [64, 197+topN*plen, 768]
        # x shape [64, 196(197), 768]
        x = self.image_encoder.ln_pre(x)            # x shape [64, 196(197), 768]
        x = x.permute(1, 0, 2)  # NLD -> LND        # x shape [196(197), 64, 768]
        x = self.image_encoder.transformer(x)       # x shape [196(197), 64, 768]
        x = x.permute(1, 0, 2)  # LND -> NLD        # x shape [64, 196(197), 768]
        x = self.image_encoder.ln_post(x[:, 0, :])  # x shape [64, 768]
        if self.image_encoder.proj is not None:
            x = x @ self.image_encoder.proj         # x shape [64, 512]
        return x
        

    ### use for prefix tuning 
    def get_image_features_prefix(self, x, prompts=None):
        x = self.image_encoder.ln_pre(x)            # x shape [64, 196(197), 768]
        x = x.permute(1, 0, 2)  # NLD -> LND        # x shape [196(197), 64, 768]
        # print('get_image_features_prefix') # the same
        # print(x)
        x, dist = self.transformer_forward_prefix(x, prompts)       # x shape [196(197), 64, 768]
        # x = self.clip.visual.transformer(x)       # x shape [196(197), 64, 768]
        # print('get_image_features_prefix') # not the same!
        # print(x)
        x = x.permute(1, 0, 2)  # LND -> NLD        # x shape [64, 196(197), 768]
        x = self.image_encoder.ln_post(x[:, 0, :])  # x shape [64, 768]
        # print('get_image_features_prefix') # not the same
        # print(x)
        if self.image_encoder.proj is not None:
            x = x @ self.image_encoder.proj         # x shape [64, 512]
        return x, dist
    
    ### use for prefix tuning
    def transformer_forward_prefix(self, x, prompts=None):
        Transformer = self.clip.visual.transformer
        dist = 0
        self.att_map = None
        # print('reset att map')
        for layer in range(Transformer.layers):
            if layer in self.image_prompt_layers:
                idx = self.image_prompt_layers.index(layer)
                prompts_one_layer = prompts[:, idx, :, :] #[batch, , plen, dim]
            else:
                prompts_one_layer = None
            # print('layer', layer)
            x = self.ResidualAttentionBlock_forward_prefix(x, layer, prompts_one_layer)
            # print('transformer_forward_prefix') # not the same
            # if layer <= 2:
            #     print(layer)
            #     print(x[0, 0, :100])
        return x, dist
    
    def get_image_features_prefix_layer(self, x, prompts=None, layers=[10, 11]):
        x = self.image_encoder.ln_pre(x)            # x shape [64, 196(197), 768]
        x = x.permute(1, 0, 2)  # NLD -> LND        # x shape [196(197), 64, 768]
        # print('get_image_features_prefix') # the same
        # print(x)
        outputs, dist = self.transformer_forward_prefix_layer(x, prompts, layers)       # x shape [196(197), 64, 768]
        
        # x = self.clip.visual.transformer(x)       # x shape [196(197), 64, 768]
        # print('get_image_features_prefix') # not the same!
        # print(x)
        layer_features = []
        patch_features = outputs[-1][1:, :, :]
        class_token_feature = outputs[-1][0, :, :].unsqueeze(0)
        for x1 in outputs:
            # print(x1.shape, 'x1 shape') # [197, 64, 768]
            x = x1.permute(1, 0, 2)  # LND -> NLD        # x shape [64, 196(197), 768]
            x = self.image_encoder.ln_post(x[:, 0, :])  # x shape [64, 768]
            # print('get_image_features_prefix') # not the same
            # print(x)
            if self.image_encoder.proj is not None:
                x = x @ self.image_encoder.proj         # x shape [64, 512]
            layer_features.append(x.unsqueeze(1))
        layer_features = torch.cat(layer_features, dim=1)
        # print(layer_features.shape, 'layer_features') # shape [64, 2, 512]
        return layer_features, dist, patch_features, class_token_feature
    
    ### use for prefix tuning
    def transformer_forward_prefix_layer(self, x, prompts=None, layers=[10, 11]): # layer num total 12
        Transformer = self.clip.visual.transformer
        dist = 0
        
        print('layer!!!')
        outputs = []
        for layer in range(Transformer.layers):
            if layer in self.image_prompt_layers:
                idx = self.image_prompt_layers.index(layer)
                prompts_one_layer = prompts[:, idx, :, :] #[batch, , plen, dim]
            else:
                prompts_one_layer = None
            x = self.ResidualAttentionBlock_forward_prefix(x, layer, prompts_one_layer)
            if layer in layers:
                outputs.append(x)
            # print('transformer_forward_prefix') # not the same
            # if layer <= 2:
            #     print(layer)
            #     print(x[0, 0, :100])
        return outputs, dist


    ### use for prefix tuning
    def ResidualAttentionBlock_forward_prefix(self, x, layer, prompts=None):
        ### Block: ResidualAttentionBlock self
        Block = self.clip.visual.transformer.resblocks[layer]
        # print('pormpts in ResidualAttentionBlock_forward_prefix')
        # print(prompts) # the same
        # print(Block)
        # input()
        q = Block.ln_1(x)
        Block.attn_mask = Block.attn_mask.to(dtype=q.dtype, device=q.device) if Block.attn_mask is not None else None
        if prompts is None:
            x = x + Block.attn(q, q, q, need_weights=False, attn_mask=Block.attn_mask)[0] #original
            # x = Block.attn(q, q, q, need_weights=False, attn_mask=Block.attn_mask)
            # # print(len(x1), 'len x1') # 2
            # print(len(x1))
            # x = x + x1[0]
            # # a = x1[1].permute(1, 0, 2)[0, :, 1:].unsqueeze(0)

            q1 = q.permute(1, 0, 2)[:, 1:, :]
            v1 = q.permute(1, 0, 2) # [support set, 197, 768]
            v1 = v1[:, 0, :].unsqueeze(1) # [support set, 1, 768]
            clique_size = q1.shape[0]
            q1 = q1.permute(0, 2, 1)
            a = torch.matmul(v1, q1)
            a = a.squeeze().unsqueeze(0)
            if clique_size == 1:
                a = a.unsqueeze(0)
            # print(a.shape)
            # print('here')
            if self.args.att_map and self.batch_learning and q.shape[1] >= 1:
                with torch.no_grad():
                    if self.att_map is None:
                        self.att_map = a.clone() # [1, support set, 196]
                    else:
                        # print(self.att_map.shape, a.shape)
                        self.att_map = torch.cat([self.att_map, a], dim=0)
            # print('ResidualAttentionBlock_forward_prefix') # not the same
            # if layer >= 10: print
            #     print(x)
            # attention_return = Block.attn(x, x, x, need_weights=False, attn_mask=Block.attn_mask)[0]
        else:
            prompts = prompts.permute(1, 0, 2)
            # print(prompts.shape)
            # input()
            half = int(prompts.shape[0]/2)
            # print(prompts[:half, :, :].unsqueeze(1).shape)
            # print(prompts[:half, :, :].shape)
            # print(prompts[:half, :, :].unsqueeze(1).shape)
            # print(prompts[:half, :, :].shape)
            # print(q[0,:,:].unsqueeze(0).shape)
            # print(q[1:,:,:].shape)
            # input()
            k = torch.cat([q[0,:,:].unsqueeze(0), prompts[:half, :, :] , q[1:,:,:]], 0)
            v = torch.cat([q[0,:,:].unsqueeze(0), prompts[half:, :, :] , q[1:,:,:]], 0)
            # print(k.requires_grad, v.requires_grad)
            # k = torch.cat([x[:,0,:].unsqueeze(1), prompts[:, :half ,:] , x[:,1:,:]], 1)
            # v = torch.cat([x[:,0,:].unsqueeze(1), prompts[:, half: ,:] , x[:,1:,:]], 1)
            # print('attention! k ,q, v: ', k.shape, q.shape, v.shape)
            # for attention map visualization:
            # if self.args.att_map and self.batch_learning and k.shape[1] >= 1:
            #     with torch.no_grad():
            #         q1 = q.permute(1, 0, 2)
            #         v1 = v.permute(1, 0, 2) # [support set, 197, 768]
            #         q1 = q1[:, 0, :].unsqueeze(1) # [support set, 1, 768]
            #         q1 = q1.permute(0, 2, 1)
            #         a = torch.matmul(v1, q1).squeeze()[:, :196].unsqueeze(0)
            #         if self.att_map is None:
            #             print('is none')
            #             self.att_map = a # [1, support set, 196]
            #         else:
            #             # print(self.att_map.shape, a.shape)
            #             self.att_map = torch.cat([self.att_map, a], dim=0)
            

            x = x + Block.attn(q, k, v, need_weights=False, attn_mask=Block.attn_mask)[0]
            # print('ResidualAttentionBlock_forward_prefix')
            # print(x)
        # x = x + attention_return
        x = x + Block.mlp(Block.ln_2(x))
        return x


    def cluster_features(self, x):
        return self.get_image_features(self.get_image_embedding(x))

    def inference(self, x, group_image_prompts=None, group_text_prompts=None):
        self.att_map = None
        if self.args.simam:
            x = self.simam(x)
        batch = x.shape[0]
        dist = 0
        # with torch.no_grad():
        #     x = self.get_image_embedding(x) # embedding shape: [batch_size, 197, 768]
            # print('image_embedding')
            # print(x)
        # image_prompts, text_prompts, dist = self.get_prompts(embedding[:,1:,:])
        ### if use image prompts
        # print('self image prompt shape: ', self.image_prompts.shape) # [1, 1, 2, 768]
        # print('self text  prompt shape: ', self.text_prompts_a.shape) # [200, 6, 512]
        image_features = self.blip_model.get_image_features(x)
        # if self.args.image_prompts:
        #     # print(self.image_prompts.repeat(24,1,1).shape)
        #     # input()
        #     ### if use Prefix Tuning
        #     if self.args.prefix_tuning:
        #         # here for A
        #         # if self.args.test_sets in ['V', 'K', 'C', 'A']:
        #         if self.use_sum:
        #             if self.args.test_sets in ['V', 'C', 'K', 'A']:
        #                 x, dist = checkpoint(self.get_image_features_prefix, x, self.image_prompts.repeat(batch,1,1,1))
        #                 # x, dist = self.get_image_features_prefix(x, self.image_prompts.repeat(batch,1,1,1))
        #             else:
        #                 # print('image_prompts')
        #                 # print(self.image_prompts.repeat(batch,1,1,1))
        #                 x, dist = self.get_image_features_prefix(x, self.image_prompts.repeat(batch,1,1,1))
        #         else:
        #             if group_image_prompts is not None:
        #                 final_image_prompts = torch.cat([self.image_prompts, group_image_prompts], dim=2)
        #             else:
        #                 final_image_prompts = self.image_prompts
        #             if self.args.parallel:
        #                 x, dist = self.get_image_features_prefix(x, final_image_prompts.repeat(batch,1,1,1))
        #             else:
        #                 x, dist = checkpoint(self.get_image_features_prefix, x, final_image_prompts.repeat(batch,1,1,1))
        #     else:    
        #         # TODO
        #         if batch == 1:
        #             x = self.get_image_features(x, self.image_prompts[0].unsqueeze(0))
        #         else:
        #             x = self.get_image_features(x, self.image_prompts)
        #     # print('image_embedding') #not the same
        #     # print(x)
        # else:
        #     # x = checkpoint(self.get_image_features, x)
        #     # with torch.no_grad():
        #     #     x = self.get_image_features(x)
        #     x = self.get_image_features(x)

        x = x / x.norm(dim=-1, keepdim=True)
        # if self.args.test_sets in ['V', 'K', 'C', 'A']:
        # if self.args.test_sets in ['V', 'K', 'C', 'A']:
        text_features = self.get_text_features(self.dummy, None)
        # if not self.args.ensemble:
        #     text_features = checkpoint(self.get_text_features, self.dummy, group_text_prompts)
        # elif self.args.learn_all:
        #     text_features = checkpoint(self.get_ensemble_text_features, self.dummy)
        # else:
        #     if self.args.parallel:
        #         text_features = self.get_text_features(self.dummy)
        #     else:
        #         text_features = checkpoint(self.get_text_features, self.dummy)
        #     # print(text_features.shape, "text_features") #[200,512]
        #     # print(self.text_ensemble_feature.shape, "ensemble features") #[200,512]
        #     # text_features1 = text_features / text_features.norm(dim=1, keepdim=True)
        #     # self.text_ensemble_feature1 = self.text_ensemble_feature / self.text_ensemble_feature.norm(dim=1, keepdim=True)
        #     # print(text_features1.T @ self.text_ensemble_feature1)
        #     if self.args.use_ensemble:
        #         text_features = (1 - self.args.ensemble_rate) * text_features + self.args.ensemble_rate * self.text_ensemble_feature.clone()

        # else:
        #     text_features = self.get_text_features()
        # text_features = self.get_text_features()
         
        # print(text_features.shape)
        # input()
        # print(text_features.norm(dim=1, keepdim=True))
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        # logits per image
        logits = logit_scale * x @ text_features.t()
        # return logits, dist
        return logits, x, text_features
    
    def inference_layer(self, x, group_image_prompts=None, layers=[10, 11]):
        batch = x.shape[0]
        dist = 0
        with torch.no_grad():
            x = self.get_image_embedding(x) # embedding shape: [batch_size, 197, 768]
            # print('image_embedding')
            # print(x)
        # image_prompts, text_prompts, dist = self.get_prompts(embedding[:,1:,:])
        ### if use image prompts
        if self.args.image_prompts:
            # print(self.image_prompts.repeat(24,1,1).shape)
            # input()
            ### if use Prefix Tuning
            if self.args.prefix_tuning:
                # here for A
                # if self.args.test_sets in ['V', 'K', 'C', 'A']:
                if self.use_sum:
                    x, dist, patch_features, class_token_feature = checkpoint(self.get_image_features_prefix_layer, x, self.image_prompts.repeat(batch,1,1,1), layers)
                    # x, dist = self.get_image_features_prefix(x, self.image_prompts.repeat(batch,1,1,1))
                else:
                    if group_image_prompts is not None:
                        final_image_prompts = torch.cat([self.image_prompts, group_image_prompts], dim=2)
                    else:
                        final_image_prompts = self.image_prompts
                    x, dist, patch_features, class_token_feature = checkpoint(self.get_image_features_prefix_layer, x, final_image_prompts.repeat(batch,1,1,1), layers)
            else:    
                # TODO
                if batch == 1:
                    x = self.get_image_features(x, self.image_prompts[0].unsqueeze(0))
                else:
                    x = self.get_image_features(x, self.image_prompts)
            # print('image_embedding') #not the same
            # print(x)
        else:
            # x = checkpoint(self.get_image_features, x)
            # with torch.no_grad():
            #     x = self.get_image_features(x)
            x = self.get_image_features(x)

        x = x / x.norm(dim=-1, keepdim=True)
        x_last = x[:, -1, :]
        print('x', x.shape)
        # if self.args.test_sets in ['V', 'K', 'C', 'A']:
        if self.args.test_sets in ['V', 'K', 'C', 'A']:
            text_features = checkpoint(self.get_text_features, self.dummy)
        else:
            text_features = self.get_text_features()
        # text_features = self.get_text_features()
         
        # print(text_features.shape)
        # input()
        # print(text_features.norm(dim=1, keepdim=True))
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        # logits per image
        logits = logit_scale * x_last @ text_features.t()
        # return logits, dist
        return logits, x, text_features, patch_features, class_token_feature
     
    def forward_clip(self, image):
        with torch.no_grad():
            image_features = self.clip.encode_image(image)
            self.tokenized_text = self.tokenized_text.to('cuda')
            # print(self.tokenized_text.device)
            text_features = self.clip.encode_text(self.tokenized_text)

            # normalized features
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)

            # cosine similarity as logits
            logit_scale = self.clip.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.t()
            # logits_per_text = logits_per_image.t()

            # shape = [global_batch_size, global_batch_size]
            return logits_per_image

    def forward(self, input, group_image_prompts=None, group_text_prompts=None, train_mode=True):
        # print(self.classnames)
        text = [f"a photo of a {classname}" for classname in self.classnames]
        # print(text)
        inputs = self.blip_processor(text=text, return_tensors="pt", padding=True)
        # print(inputs.input_ids.shape)
        text_features = self.blip_model.get_text_features(**inputs)
        image_features = self.blip_model.get_image_features(input)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        return logits, image_features, text_features
        # return self.inference(input, group_image_prompts=group_image_prompts, group_text_prompts=group_text_prompts)



class SimamModule(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-3):
        super(SimamModule, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.activaton(y)

# get_coop(args, args.arch, args.test_sets, args.gpu, args.n_ctx, args.ctx_init)
def get_coop(args, clip_arch, test_set, device, n_ctx, ctx_init, learned_cls=False):
    if test_set in fewshot_datasets:
        classnames = eval("{}_classes".format(test_set.lower()))
    elif test_set == 'bongard':
        if learned_cls:
            classnames = ['X', 'X']
        else:
            classnames = ['True', 'False']
    else:
        classnames = imagenet_classes
    # print(len(classnames))
# args, device, classnames, batch_size, 
#         criterion='cosine', arch="ViT-L/14"
    if args.use_blip:
        model = BlipTestTimeTuning(args, device, classnames, arch=clip_arch)
    else:
        model = ClipTestTimeTuning(args, device, classnames, arch=clip_arch)

    return model
