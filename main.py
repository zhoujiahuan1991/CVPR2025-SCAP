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
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import BlipProcessor, BlipModel, AutoProcessor
import os
from fvcore.nn import FlopCountAnalysis


try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import torchvision.models as models

from clip.my_clip_CSTP import get_coop
from clip.cocoop import get_cocoop
from data.imagnet_prompts import imagenet_classes
from data.datautils import AugMixAugmenter, build_dataset, get_config_file, build_test_data_loader, build_test_dataset
from utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, load_model_weight, set_random_seed
from data.cls_to_names import *
from data.fewshot_datasets import fewshot_datasets
from data.imagenet_variants import thousand_k_to_200, imagenet_a_mask, imagenet_r_mask, imagenet_v_mask, tiny_imagenet_mask

from utils.aug_tools import AugTools
from utils.ema import Text_EMA, Image_EMA
from utils.tda.utils import *

from torchvision.datasets import CIFAR100

from feature_prompt import add_feature_prompt
from feature_prompt import update_feature_prompt
from feature_prompt import update_class_prompt
from feature_prompt import update_all_class_prompt
from batch_utils import BatchLearner, save_images_to_folder
from feature_cache import FeatureCache

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

feature_prompt_cache = []
class_prompt_cache = []
class_prompt_init = []
pos_cache, neg_cache = {}, {}
pos_ema_cache = {}
neg_ema_cache = {}

 
def break_sample_tie(ties, logit, device):
    ties = torch.tensor(ties, dtype=torch.int, device=device)
    logit[~ties] = -torch.inf
    scalar_pred = torch.argmax(logit, dim=-1)
    return scalar_pred


def greedy_break(ties, logits, device):
    ties_tensor = torch.tensor(ties, dtype=torch.int, device=device)
    preds = torch.argmax(logits, dim=1)
    for pred in preds:
        if pred in ties_tensor:
            return pred
    return break_sample_tie(ties, logit=logits[0], device=device)

def select_confident_samples(logits, top):
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * top)]
    return logits[idx], idx

def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True) # logits = outputs.log_softmax(dim=1) [N, 1000]
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0]) # avg_logits = logits.mean(0) [1, 1000]
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)


def test_time_tuning(model, inputs, optimizer, scaler, args, pos_params=None, neg_params=None, target=None, class_num=None, group_image_prompts=None):
    global pos_cache
    global neg_cache
    global pos_ema_cache
    global neg_ema_cache
    if args.cocoop:
        image_feature, pgen_ctx = inputs
        pgen_ctx.requires_grad = True
        optimizer = torch.optim.AdamW([pgen_ctx], args.lr)
    
    # if args.parallel:
    #     model = nn.DataParallel(model)
    
    for j in range(args.tta_steps):
        with torch.cuda.amp.autocast():
            # flops = FlopCountAnalysis(model, inputs)
            # print(flops.total()/1e9)
            output, image_features, _ = model(inputs, group_image_prompts=group_image_prompts) # add group prompt or not??
            if j == 0:
                global aug_tools
                aug_tools.cal_clip(output[0])
            selected_idx = None
            if selected_idx is not None:
                output = output[selected_idx]
            else:
                output, selected_idx = select_confident_samples(output, args.selection_p)
            image_features = image_features[selected_idx].mean(0).unsqueeze(0)
            aug_tools.cal_aug(output, j)
            loss = avg_entropy(output)
            avg_output = output.mean(0).unsqueeze(0)
            prob_map = output.softmax(1).mean(0).unsqueeze(0)
            pred = int(avg_output.topk(1, 1, True, True)[1].t())

            final_logits = avg_output
            final_loss = loss
            weight = avg_output.max(dim=1, keepdim=True)[0].squeeze().item()

            # TDA
            # Firstly, update TDA cache (only once)
            with torch.no_grad():
                if args.pos_enabled and j == 0:
                    update_cache(pos_cache, pred, [image_features, loss], pos_params['shot_capacity'], target=target)
                if args.neg_enabled and j == 0 and neg_params['entropy_threshold']['lower'] < loss / 6.768184324776926 < neg_params['entropy_threshold']['upper']:
                    update_cache(neg_cache, pred, [image_features, loss, prob_map], neg_params['shot_capacity'], True)
                if args.pos_ema_enabled and j == 0 and neg_params['entropy_threshold']['lower'] < loss / 6.768184324776926 < neg_params['entropy_threshold']['upper']:
                    update_ema_cache(pos_ema_cache, pred, [image_features, loss], pos_params['shot_capacity'], target=target, mode=args.ema_mode, weight=weight, h=args.ema_h)
            
            # Secondly, if enable 1, apply TDA inside prompt tuning
            if args.enable1:
                if args.pos_enabled and pos_cache:
                    pos_logits, pos_mask = compute_cache_logits(image_features, pos_cache, pos_params['alpha'], pos_params['beta'], len(model.classnames))
                    final_logits += pos_logits
                if args.neg_enabled and neg_cache:
                    neg_logits, neg_mask = compute_cache_logits(image_features, neg_cache, neg_params['alpha'], neg_params['beta'], len(model.classnames), (neg_params['mask_threshold']['lower'], neg_params['mask_threshold']['upper']))
                    final_logits -= neg_logits
                if args.pos_ema_enabled and pos_ema_cache:
                    pos_ema_logits, pos_ema_mask = compute_cache_logits(image_features, pos_ema_cache, pos_params['alpha'], pos_params['beta'], len(model.classnames))
                    final_logits = final_logits + pos_ema_logits * args.ema_weight
                
                final_loss = avg_entropy(final_logits)

                if args.pos_dropout:
                    dropout(pos_cache, args.pos_dropout_rate)
                if args.neg_dropout:
                    dropout(neg_cache, args.neg_dropout_rate)
                    
            # Compute the loss of mean and var parameters (not used in the final version)
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    test_mean = module.mean
                    test_var = module.var
                    final_loss += torch.abs(test_mean - R_M[name]).sum() + torch.abs(test_var - R_V[name]).sum()
        
        optimizer.zero_grad()
        # compute gradient and do SGD step
        scaler.scale(final_loss).backward()

        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.grad)
        # Unscales the gradients of optimizer's assigned params in-place
        scaler.step(optimizer)
        scaler.update()
        with torch.cuda.amp.autocast():
            if args.cocoop:
                output = model((image_feature, pgen_ctx))
            else:
                output, _, _ = model(inputs) 
            output, selected_idx = select_confident_samples(output, args.selection_p)
            aug_tools.cal_trained_aug(output, j)
    output_1 = torch.mean(output, dim=0)
    if args.cocoop:
        return output, pgen_ctx
    # if args.parallel:
    #     model = model.module
    return output_1, output

def subset_prompt_learning(model, inputs, optimizer, scaler, args):
    for j in range(args.tta_steps):
        with torch.cuda.amp.autocast():
            output, _, _ = model(inputs) 
            loss = avg_entropy(output)
        
        optimizer.zero_grad()
        # compute gradient and do SGD step
        scaler.scale(loss).backward()
        # Unscales the gradients of optimizer's assigned params in-place
        scaler.step(optimizer)
        scaler.update()
    return

aug_tools = None

def main():
    args = parser.parse_args()
    print(args.ctx_init, args.n_ctx)
    set_random_seed(args.seed)
    global aug_tools
    aug_tools = AugTools(args)
    aug_tools.logger.info(f'Threshold: {args.image_feature_threshold}, Template: {args.ctx_init}')
    aug_tools.logger.info(f'Limit of graph: {args.limit}, Degree: {args.degree}, Graph Alpha: {args.graph_alpha}')
    aug_tools.logger.info(f'w_step: {args.w_step}, w_prompt: {args.w_prompt}, w_pow: {args.w_pow}')

    # This codebase has only been tested under the single GPU setting
    assert args.gpu is not None
    main_worker(args.gpu, args)

    
R_M = {}
R_V = {}
I_F = None
Label = None

def main_worker(gpu, args):
    args.gpu = gpu
    set_random_seed(args.seed)
    print("Use GPU: {} for training".format(args.gpu))

    # create model (zero-shot clip model (ViT-L/14@px336) with promptruning)
    if args.test_sets in fewshot_datasets:
        classnames = eval("{}_classes".format(args.test_sets.lower()))
    else:
        classnames = imagenet_classes
    if args.myclip:
        model = get_coop(args, args.arch, args.test_sets, args.gpu, args.n_ctx, args.ctx_init)
        # if args.parallel:
        #     model = DataParallel(model)
    model_state = None
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            R_M[name] = module.running_mean.clone()
            R_V[name] = module.running_var.clone()

    for name, param in model.named_parameters():
        # print(name)
        if args.myclip:
            if name == "image_prompts" and args.image_prompts:
                param.requires_grad_(True)
            elif "prompt_transformer" not in name:
                param.requires_grad_(False)
        elif not args.cocoop:
            if "prompt_learner" not in name:
                param.requires_grad_(False)
        else:
            if "text_encoder" not in name:
                param.requires_grad_(False)
    
    print("=> Model created: visual backbone {}".format(args.arch))
    
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    else:
        assert args.gpu is not None
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    # define optimizer
    if args.cocoop:
        optimizer = None
        optim_state = None
    elif args.myclip:
        trainable_param = []
    else:
        trainable_param = model.prompt_learner.parameters()
        optimizer = torch.optim.AdamW(trainable_param, args.lr)
        optim_state = deepcopy(optimizer.state_dict())

    # setup automatic mixed-precision (Amp) loss scaling
    scaler = torch.cuda.amp.GradScaler(init_scale=1000)

    print('=> Using native Torch AMP. Training in mixed precision.')

    cudnn.benchmark = True

    # norm stats from clip.load()
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])

    
    # iterating through eval datasets
    datasets = args.test_sets.split("/")
    results = {}
    for set_id in datasets:
        if args.tpt:
            base_transform = transforms.Compose([
                    transforms.Resize(args.resolution, interpolation=BICUBIC),
                    transforms.CenterCrop(args.resolution)])
            # if args.resize_flag is True:
            #     base_transform = transforms.Compose([
            #         transforms.Resize(args.resize, interpolation=BICUBIC),
            #         transforms.CenterCrop(args.resolution)])
            # else:
            #     base_transform = transforms.Compose([
            #         transforms.Resize(args.resolution, interpolation=BICUBIC),
            #         transforms.CenterCrop(args.resolution)])
            preprocess = transforms.Compose([
                transforms.ToTensor(),
                normalize])
            # data_transform = AugMixAugmenter(base_transform, preprocess, n_views=args.batch_size-1, 
            #                                 augmix=len(set_id)>1, args=args)
            data_transform = AugMixAugmenter(base_transform, preprocess, n_views=args.batch_size-1, 
                                            augmix=False, args=args)
            batchsize = args.batch_num

        print("evaluating: {}".format(set_id))
        cd = False
        if set_id in ['sun397', 'ucf101', 'caltech101', 'dtd', 'eurosat', 'fgvc', 'food101', 'oxford_flowers', 'oxford_pets', 'stanford_cars', 'ucf101', 'stanford_cars_old']:
            # cfg = get_config_file('configs', set_id)
            cd = True
            val_dataset = None
            test_loader, classnames, template = build_test_data_loader(set_id, args.data_root, data_transform, batchsize)
            # if set_id in ['sun397']:
            #     print(classnames)
        elif set_id in ['A', 'R', 'K', 'V', 'I', 'C', 'T']:
            classnames_all = imagenet_classes
            classnames = []
            if set_id in ['A', 'R', 'V']:
                label_mask = eval("imagenet_{}_mask".format(set_id.lower()))
                if set_id == 'R':
                    for i, m in enumerate(label_mask):
                        if m:
                            classnames.append(classnames_all[i])
                else:
                    classnames = [classnames_all[i] for i in label_mask]
            else:
                classnames = classnames_all
        
        print(classnames)


        if args.cocoop:
            model.prompt_generator.reset_classnames(classnames, args.arch)
            model = model.cpu()
            model_state = model.state_dict()
            model = model.cuda(args.gpu)
        elif args.myclip:
            model.init_text_prompts(classnames)
            print("learnable text: ", args.learnable_text)
            if args.CSTP == 1:
                if args.learnable_text == "a":
                    trainable_param += [model.text_prompts_a]
                    # print("learnable text: ", args.learnable_text)
                    # input()
                elif args.learnable_text == "a+cls":
                    trainable_param += [model.text_prompts_a]
                    trainable_param += [model.text_prompts_class]
                elif args.learnable_text == "S+a+cls+E":
                    trainable_param += [model.text_prompts_S]
                    trainable_param += [model.text_prompts_a]
                    trainable_param += [model.text_prompts_class]
                    trainable_param += [model.text_prompts_E]
                elif args.learnable_text == "all":
                    trainable_param += [model.text_prompts]
            elif args.CSTP == 2:
                if args.learnable_text == "a":
                    trainable_param += [model.CSTP_bvector]
                    trainable_param += [model.text_prompts_a]
            if args.image_prompts:
                trainable_param += [model.image_prompts]
            elif args.prompt_pool:
                trainable_param += [model.prompt_pool.keys]
                trainable_param += [model.prompt_pool.img_prompts]
            optimizer = torch.optim.AdamW(trainable_param, args.lr)
            optim_state = deepcopy(optimizer.state_dict())
        else:
            model.reset_classnames(classnames, args.arch)
        
        if not cd:
            val_dataset = build_dataset(set_id, data_transform, args.data, mode=args.dataset_mode, domain_id=args.domain)
            print("number of test samples: {}".format(len(val_dataset)))
            val_loader = torch.utils.data.DataLoader(
                        val_dataset,
                        batch_size=batchsize, shuffle=True,
                        num_workers=args.workers, pin_memory=True)
        else:
            val_loader = test_loader
            
        # 
        if args.text_prompt_ema:
            if args.myclip:
                text_ema = Text_EMA("text_prompts_a", model.text_prompts_a, args, args.text_prompt_ema_decay)
            else:
                text_ema = Text_EMA("prompt_learner.ctx", model.prompt_learner.ctx, args)
        else:
            text_ema = None
        if args.image_prompt_ema == 1 or args.image_prompt_ema == 2:
            image_ema = Image_EMA("image_prompts", model.image_prompts, args, args.image_prompt_ema_decay)
        elif args.image_prompt_ema == 3 or args.image_prompt_ema == 4:
            class_image_prompts = model.image_prompts.unsqueeze(0).repeat(len(classnames), 1, 1, 1, 1)
            image_ema = Image_EMA("image_prompts", class_image_prompts, args, args.image_prompt_ema_decay)
        else:
            image_ema = None

        # set up class prompts
        for i in range(len(classnames)):
            class_prompt_cache.append(model.image_prompts.clone())
            class_prompt_init.append(model.image_prompts.clone())

        pos_params = {'shot_capacity': args.pos_shot_capacity, 'alpha': args.pos_alpha, 'beta': args.pos_beta}
        neg_params = {'shot_capacity': args.neg_shot_capacity, 'alpha': args.neg_alpha, 'beta': args.neg_beta, 'entropy_threshold': {'lower': 0.2, 'upper': 0.5}, 'mask_threshold': {'lower': 0.03, 'upper': 1.0}}
            
        test_time_adapt_eval(val_loader, model, model_state, optimizer, optim_state, scaler, args, classnames, text_ema, image_ema, pos_params, neg_params)
        del val_dataset, val_loader



def test_time_adapt_eval(val_loader, model, model_state, optimizer, optim_state, scaler, args, classnames, text_ema=None, image_ema=None, pos_params=None, neg_params=None):

    # reset model and switch to evaluate mode
    model.eval()
    if not args.cocoop: # no need to reset cocoop because it's fixed
        with torch.no_grad():
            ## TODO
            model.reset()
    batch_learner = BatchLearner(len(classnames), args, model)
    # print(model.text_prompts_a.shape) # [200, 6, 512]

    global aug_tools
    global class_prompt_cache
    global pos_ema_cache
    global neg_ema_cache
    global I_F
    global Label
    

    for i, (images, target) in enumerate(val_loader):

        assert args.gpu is not None
        if isinstance(images, list):
            for k in range(len(images)):
                images[k] = images[k].cuda(args.gpu, non_blocking=True)
            image = images[0]
        else:
            if len(images.size()) > 4:
                # when using ImageNet Sampler as the dataset
                assert images.size()[0] == 1
                images = images.squeeze(0)
            images = images.cuda(args.gpu, non_blocking=True)
            image = images
        target = target.cuda(args.gpu, non_blocking=True)


        if args.tpt:
            for j in range(len(images)):
                images[j] = images[j].unsqueeze(0)
                # print(images[j].shape)
            images = torch.cat(images, dim=0)

        # print(images.shape)
        # # [aug_num, b, 3, 224, 224]
        aug_num, batch_num, _, _, _ = images.shape

        with torch.no_grad():
            # outputs, image_features_layer, text_features, patch_features, class_token_feature = model.inference_layer(images[0, :], layers=[10, 11])
            outputs, image_features, text_features = model(images[0, :])
            if args.orderly:
                confidences = outputs.max(dim=1)
                order = torch.argsort(confidences)
                print(order)
        
        if args.use_group:
            batch_learner.get_subgroups(image_features, outputs)
            batch_learner.subgroups_prompt_tuning(model, images, optimizer, scaler, args, image_features, outputs, text_ema, classnames)
            batch_learner.update_dart_text_ema(text_ema)


        for j in range(images.shape[1]):
            # reset the tunable prompt to its initial state
            if args.myclip:
                with torch.no_grad():
                    model.reset_Tclass_prompts()
                    if args.reset_image_prompts:
                        model.reset_image_prompts()
                    if args.share_prompts != 0 and args.reset_share_prompts:
                        model.reset_share_prompts()
            elif args.tta_steps > 0:
                with torch.no_grad():
                    model.reset()
            if args.text_prompt_ema:
                new_param = text_ema.apply_shadow("text_prompts_a", 
                                                                model.text_prompts_a, 
                                                                w=args.text_prompt_ema_w)
                model.state_dict()["text_prompts_a"].copy_(new_param)
            optimizer.load_state_dict(optim_state)
            single_target = target[j]
            aug_tools.target = single_target
            
            if True:
                output_1, output_aug = test_time_tuning(model, images[:, j], optimizer, scaler, args, pos_params, neg_params, single_target.item(), len(model.classnames))
            
                output_tmp = output_1.unsqueeze(0)
                pred_class = output_tmp.argmax(dim=1, keepdim=True).squeeze()
                weight = output_tmp.max(dim=1, keepdim=True)[0].squeeze()
                new_param_image = image_ema.apply_shadow_one("image_prompts", 
                                                        model.image_prompts,
                                                        pred_class, 
                                                        w=args.image_prompt_ema_w)
                
                if args.use_group:
                    group_image_prompts, group_text_prompts_ = batch_learner.get_prompt(image_features[j].unsqueeze(0), j, args, pred_class, retention=args.use_retention)
                else:
                    group_image_prompts = None
                model.state_dict()["image_prompts"].copy_(new_param_image)

                group_text_prompts = None
                
            
            
            # The actual inference goes here
            if args.tpt:
                if args.cocoop:
                    image_feature = image_feature[0].unsqueeze(0)       
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    if args.use_sum:
                        output, image_feature, text_features = model(images[0, j].unsqueeze(0))
                    else:
                        output, image_feature, text_features = model(images[0, j].unsqueeze(0), group_image_prompts=None, group_text_prompts=None)
                        
                    loss = avg_entropy(output)
                    final_logits = output
                    if args.enable2 and (loss / 6.768184324776926) > 0:
                            if args.pos_enabled and pos_cache:
                                # noised_pos_cache = get_noised_pos_cache(pos_cache, scale=0.0)
                                pos_logits, pos_mask = compute_cache_logits(image_feature, pos_cache, pos_params['alpha'], pos_params['beta'], len(model.classnames))
                                final_logits = final_logits + pos_logits
                            if args.neg_enabled and neg_cache:
                                neg_logits, neg_mask = compute_cache_logits(image_feature, neg_cache, neg_params['alpha'], neg_params['beta'], len(model.classnames), (neg_params['mask_threshold']['lower'], neg_params['mask_threshold']['upper']))
                                final_logits = final_logits - neg_logits
                            if args.pos_ema_enabled and pos_ema_cache:
                                pos_ema_logits, pos_ema_mask = compute_cache_logits(image_feature, pos_ema_cache, pos_params['alpha'], pos_params['beta'], len(model.classnames))
                                final_logits = final_logits + pos_ema_logits * args.ema_weight
                    output = final_logits
            aug_tools.cal_trained(output)
            aug_tools.cal_acc(output)

            pred_class = output.argmax(dim=1, keepdim=True).squeeze().item()

            if args.save_samples and (single_target != pred_class):
                save_images_to_folder(images[0, j].unsqueeze(0), 'wrong_samples/Targ_{}/Targ_{},Pred_{}'.format(classnames[single_target], classnames[single_target], classnames[pred_class]))

            weight = output.max(dim=1, keepdim=True)[0].squeeze().item()
            score = output.softmax(1)
            entropy = -(score * torch.log(score)).sum()

            if args.text_prompt_ema:
                if args.text_prompt_ema_weight:
                    text_ema.update_weight("text_prompts_a", model.text_prompts_a, output.squeeze(),
                                            args.text_prompt_ema_weight_h)
                elif args.text_prompt_ema_one:
                    text_ema.update_one("text_prompts_a", model.text_prompts_a, pred_class)
                elif args.text_prompt_ema_one_weight:
                    # print(model.text_prompts_a.shape, '530, text prompts') [200, 6, 512]
                    text_ema.update_one_weight("text_prompts_a", model.text_prompts_a, 
                                                pred_class, weight,
                                                args.text_prompt_ema_one_weight_h)
                else:
                    text_ema.update("text_prompts_a", model.text_prompts_a)
            if args.image_prompt_ema == 1:
                image_ema.update("image_prompts", model.image_prompts)
            elif args.image_prompt_ema == 2:
                image_ema.update_weight("image_prompts", model.image_prompts, weight, args.image_prompt_ema_h)
            elif args.image_prompt_ema == 3:
                image_ema.update_one("image_prompts", model.image_prompts, pred_class)
            elif args.image_prompt_ema == 4:
                image_ema.update_one_weight("image_prompts", model.image_prompts, pred_class, weight, args.image_prompt_ema_h)

        if (i+1) % args.print_freq == 0:
            print(args.enable_aug)
            aug_tools.logger_nums(aug=args.enable_aug)
        
    aug_tools.logger_nums(end=True, aug=args.enable_aug)

    return 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test-time Prompt Tuning')
    parser.add_argument('--myclip', action='store_true', default=False, help="")
    parser.add_argument('data', metavar='DIR', help='path to dataset root')
    parser.add_argument('--data_root', type=str, default='/data/dataset/zhangchenyu/data', help='path to cd datasets')
    parser.add_argument('--test_sets', type=str, default='A/R/V/K/I', help='test dataset (multiple datasets split by slash)')
    parser.add_argument('--dataset_mode', type=str, default='test', help='which split to use: train/val/test')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='ViT-B/16')
    # parser.add_argument('-a', '--arch', metavar='ARCH', default='RN50')
    parser.add_argument('--resize', default=256, type=int, help='CLIP image resolution')
    parser.add_argument('--resize_flag', default=False, type=bool, help='CLIP image resolution')
    parser.add_argument('--resolution', default=224, type=int, help='CLIP image resolution')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch_size', default=64, type=int, metavar='N')
    parser.add_argument('--lr', '--learning-rate', default=5e-3, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('-p', '--print-freq', default=1, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use.')
    parser.add_argument('--tpt', action='store_true', default=False, help='run test-time prompt tuning')
    parser.add_argument('--selection_p', default=0.1, type=float, help='confidence selection percentile')
    parser.add_argument('--tta_steps', default=1, type=int, help='test-time-adapt steps')
    parser.add_argument('--n_ctx', default=4, type=int, help='number of tunable tokens')
    parser.add_argument('--ctx_init', default="a_photo_of_a_CLS", type=str, help='init tunable prompts')
    parser.add_argument('--cocoop', action='store_true', default=False, help="use cocoop's output as prompt initialization")
    parser.add_argument('--load', default=None, type=str, help='path to a pre-trained coop/cocoop')
    parser.add_argument('--seed', type=int, default=19)
    parser.add_argument('--break_m', type=int, default=-1)
    parser.add_argument('--info', type=str, default='debugg')
    # the type of augmentation
    parser.add_argument('--aug_type', type=str, default='default')
    parser.add_argument('--CSTP', type=int, default=1)
    parser.add_argument('--CSTP_N', type=int, default=200)
    parser.add_argument('--text_prompt_ema', action='store_true', default=False)
    parser.add_argument('--text_prompt_ema_weight', action='store_true', default=False)
    parser.add_argument('--text_prompt_ema_weight_h', type=float, default=37)
    parser.add_argument('--text_prompt_ema_one', action='store_true', default=False)
    parser.add_argument('--text_prompt_ema_one_weight', action='store_true', default=False)
    parser.add_argument('--text_prompt_ema_one_weight_h', type=float, default=37)
    parser.add_argument('--text_prompt_ema_w', type=float, default=0.5)
    parser.add_argument('--text_prompt_ema_decay', type=float, default=0.99)
    parser.add_argument('--learnable_text', type=str, default="a")
    
    parser.add_argument('--image_prompts', action='store_true', default=False, help="")
    parser.add_argument('--prefix_tuning', action='store_true', default=True, help="Using Prefix Tuning if True, \
                                                                    Prompt Tuning if False")
    parser.add_argument('--reset_image_prompts', action='store_true', default=False, help="")
    parser.add_argument('--image_prompt_layer', nargs="+", type=int, default=[1])
    
    parser.add_argument('--image_prompt_ema', type=float, default=0, help="")
    parser.add_argument('--image_prompt_ema_decay', type=float, default=0.995, help="")
    parser.add_argument('--image_prompt_ema_w', type=float, default=0.5, help="")
    parser.add_argument('--image_prompt_ema_h', type=float, default=5000, help="")
    parser.add_argument('--share_prompts', type=int , default=0)
    parser.add_argument('--prompt_pool', action='store_true', default=False, help="use prompt pool")
    
    parser.add_argument('--domain', type=str, default="brightness")

    # new module
    parser.add_argument('--use_sum', type=int, default=0)
    parser.add_argument('--threshold', type=float , default=0.85)
    parser.add_argument('--update_threshold', type=float, default=0.9)
    parser.add_argument('--similar_threshold', type=float, default=0.9)
    parser.add_argument('--group_threshold', type=float, default=0.9)
    parser.add_argument('--max_length', type=int, default=5)
    parser.add_argument('--retention_prompt_select_num', type=int, default=1)
    parser.add_argument('--min_length_for_retention', type=int, default=3)
    parser.add_argument('--group_update_dart_rate', type=float, default=0.001)
    parser.add_argument('--batch_num', type=int, default=64)
    parser.add_argument('--transfer', type=int, default=1)
    parser.add_argument('--image_feature_threshold', type=float, default=0.8)
    parser.add_argument('--use_group', type=int, default=1)
    parser.add_argument('--use_retention', type=int, default=1)
    parser.add_argument('--retention_rate', type=int, default=100)
    parser.add_argument('--orderly', type=int, default=0)

    parser.add_argument('--pos_enabled', type=int, default=0, help='1 for positive cache enabled and 0 for not')
    parser.add_argument('--neg_enabled', type=int, default=0, help='1 for negative cache enabled and 0 for not')
    parser.add_argument('--pos_shot_capacity', type=int, default=3, help='The shot capacity refers to the maximum number of pairs per class')
    parser.add_argument('--neg_shot_capacity', type=int, default=2, help='The shot capacity refers to the maximum number of pairs per class')
    parser.add_argument('--pos_alpha', type=float, default=2.9, help='hyperparameter for the Tip-adapter part of TDA')
    parser.add_argument('--neg_alpha', type=float, default=0.22, help='hyperparameter for the Tip-adapter part of TDA')
    parser.add_argument('--pos_beta', type=float, default=8.0)
    parser.add_argument('--neg_beta', type=float, default=1.0)
    parser.add_argument('--enable1', type=int, default=0) # 1
    parser.add_argument('--enable2', type=int, default=0) # 1
    parser.add_argument('--pos_dropout', type=int, default=1) # 1
    parser.add_argument('--pos_dropout_rate', type=float, default=0.1)
    parser.add_argument('--neg_dropout', type=int, default=1) # 1
    parser.add_argument('--neg_dropout_rate', type=float, default=0.01)

    parser.add_argument('--reset_to_retention', type=int, default=1)
    parser.add_argument('--text_w', type=float, default=0.9)
    parser.add_argument('--text_group_w', type=float, default=0.0)

    parser.add_argument('--text_prompt_aggregation', type=int, default=0)
    parser.add_argument('--text_feature_aggregation', type=int, default=0)
    parser.add_argument('--text_logits_aggregation', type=int, default=0)

    parser.add_argument('--w_step', type=float, default=1.0)
    parser.add_argument('--w_prompt', type=float, default=0.1)
    parser.add_argument('--w_pow', type=float, default=1.0)

    parser.add_argument('--pos_ema_enabled', type=int, default=1) # 1
    parser.add_argument('--ema_mode', type=str, default='dart', help='pick from: dart, fix')
    parser.add_argument('--ema_weight', type=float, default=3.0)
    parser.add_argument('--ema_h', type=int, default=5000)

    parser.add_argument('--ensemble', type=int, default=0)
    parser.add_argument('--mse_loss_weight', type=float, default=0.01)
    parser.add_argument('--limit', type=int, default=5)
    parser.add_argument('--degree', type=int, default=2)
    parser.add_argument('--graph_alpha', type=float, default=0.3)

    parser.add_argument('--learn_all', type=int, default=0)
    parser.add_argument('--use_ensemble', type=int, default=1)
    parser.add_argument('--ensemble_rate', type=float, default=0.5)

    parser.add_argument('--save_samples', type=int, default=0)
    parser.add_argument('--save_groups', type=int, default=0)
    parser.add_argument('--simam', type=int, default=0)
    parser.add_argument('--batch_aug', type=int, default=0)
    parser.add_argument('--att_map', type=int, default=0)
    parser.add_argument('--parallel', type=int, default=0)
    parser.add_argument('--use_blip', type=int, default=0)
    
    parser.add_argument("--enable_aug", type=int, default=0)



    main()