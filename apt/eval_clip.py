import os
import torch
import yaml
import argparse
from warnings import warn
from yacs.config import CfgNode
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np

from statistics import mean

from torch import Tensor
import torch.nn as nn
from torch.autograd import grad, Variable
from torchvision import transforms
from torchvision.datasets import *

from collections import OrderedDict
from typing import Tuple, TypeVar

from addict import Dict

from dassl.data import DataManager

from datasets import (
    oxford_pets, oxford_flowers, fgvc_aircraft, dtd, eurosat, 
    stanford_cars, food101, sun397, caltech101, ucf101, imagenet
)

from torchattacks import PGD, TPGD
from autoattack import AutoAttack

from utils import *
from clip.simple_tokenizer import SimpleTokenizer

def CWLoss(output, target, confidence=0):
    """
    CW loss (Margin loss).
    """
    num_classes = output.shape[-1]
    target = target.data
    target_onehot = torch.zeros(target.size() + (num_classes,))
    target_onehot = target_onehot.cuda()
    target_onehot.scatter_(1, target.unsqueeze(1), 1.)
    target_var = Variable(target_onehot, requires_grad=False)
    real = (target_var * output).sum(1)
    other = ((1. - target_var) * output - target_var * 10000.).max(1)[0]
    loss = - torch.clamp(real - other + confidence, min=0.)
    loss = torch.sum(loss)
    return loss

def input_grad(imgs, targets, model, criterion):
    output = model(imgs)
    loss = criterion(output, targets)
    ig = grad(loss, imgs)[0]
    return ig

def perturb(imgs, targets, model, criterion, eps, eps_step, pert=None, ig=None):
    adv = imgs.requires_grad_(True) if pert is None else torch.clamp(imgs+pert, 0, 1).requires_grad_(True)
    ig = input_grad(adv, targets, model, criterion) if ig is None else ig
    if pert is None:
        pert = eps_step*torch.sign(ig)
    else:
        pert += eps_step*torch.sign(ig)
    pert.clamp_(-eps, eps)
    adv = torch.clamp(imgs+pert, 0, 1)
    pert = adv-imgs
    return adv.detach(), pert.detach()

def pgd(imgs, targets, model, criterion, eps, eps_step, max_iter, pert=None, ig=None):
    for i in range(max_iter):
        adv, pert = perturb(imgs, targets, model, criterion, eps, eps_step, pert, ig)
        ig = None
    return adv, pert

def load_clip_to_cpu(backbone_name="RN50"):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model

parser = argparse.ArgumentParser()
parser.add_argument('experiment', type=str, help="Name or type of experiment to run.")

parser.add_argument('-cp', '--cls-prompt', type=str, default='a photo of a {}', 
                    help="Template for class prompt. Default is 'a photo of a {}'.")

parser.add_argument('-ap', '--atk-prompt', type=str, default=None, 
                    help="Template for attack prompt. If not specified, defaults to None.")

parser.add_argument('--best-checkpoint', action='store_true',
                    help="Use the best checkpoint if available.")

parser.add_argument('--attack', type=str, default='pgd', 
                    help="Type of attack to use. Default is 'pgd'.")

parser.add_argument('--dataset', type=str, default=None, 
                    help="Dataset to use for the experiment. Defaults to None.")

parser.add_argument('-lp', '--linear-probe', action='store_true', 
                    help="Enable linear probing for the experiment.")

parser.add_argument('--save-img', action='store_true', 
                    help="Enable save images for the experiment.")

parser.add_argument('--save-path', type=str, default = './',
                    help="Specific path to save images. Default is ./")

parser.add_argument('--num-imgs', type=int, default = '10',
                    help="Number of images to save. Default is 10")

parser.add_argument('--seed', type=int, default = '42',
                    help="Seed for torch random")

parser.add_argument("--topk", type=int, default = '1',
                    help="Select top-k similar words")

if __name__ == '__main__':
    args = parser.parse_args()
    cfg = CfgNode()
    cfg.set_new_allowed(True)
    cfg_path = os.path.join(args.experiment, 'cfg.yaml')
    cfg.merge_from_file(cfg_path)

    train_dataset = cfg.DATASET.NAME
    os.makedirs(args.save_path, exist_ok=True)
    if args.dataset:
        if args.dataset in ['ImageNetR', 'ImageNetA', 'ON']:
            cfg.DATASET.NAME = 'ImageNet'
        else:
            cfg.DATASET.NAME = args.dataset
        save_path = os.path.join(args.save_path, 'dist_shift.yaml')  # Modified to use args.save_path
    else:
        save_path = os.path.join(args.save_path, 'evaluation.yaml')  # Modified to use args.save_path
    if os.path.isfile(save_path):
        with open(save_path, 'r') as f:
            result = Dict(yaml.safe_load(f))

        result = result if args.dataset is None or args.dataset==train_dataset else result[args.dataset]
        tune = 'linear_probe' if args.linear_probe else args.cls_prompt
        if result[tune][args.attack] != {}:
            print(f'eval result already exists at: {save_path}')
            exit()
            
    dm = DataManager(cfg)
    classes = dm.dataset.classnames
    loader = dm.test_loader
    num_classes = dm.num_classes

    if args.dataset in ['ImageNetR', 'ImageNetA', 'ON'] or (train_dataset == 'ImageNet' and args.dataset is None and args.attack == 'aa'):
        from OODRB.imagenet import ImageNet
        if args.dataset == 'ImageNetV2':
            shift = 'v2'
        elif args.dataset == 'ImageNetA':
            shift = 'A'
        elif args.dataset == 'ImageNetR':
            shift = 'R'
        elif args.dataset == 'ON':
            shift = 'ON'
        else:
            shift = None
        num_classes = 1000
        dataset = ImageNet(cfg.DATASET.ROOT,
                           shift,
                           'val',
                           transform=loader.dataset.transform)
        if args.attack == 'aa':
            dataset = torch.utils.data.Subset(dataset, list(range(5000)))
        loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=100,
                                             shuffle=False,
                                             num_workers=4,
                                             pin_memory=True)

    model, _ = clip.load(cfg.MODEL.BACKBONE.NAME, device='cpu')

    # load pretrained adversarially robust backbone models
    # ckp_name = 'vitb32' if cfg.MODEL.BACKBONE.NAME == 'ViT-B/32' else 'rn50'
    # eps = int(cfg.AT.EPS * 255)
    # # eps = 
    # ckp_name += f'_eps{eps}.pth.tar'
    # ckp = torch.load(os.path.join('backbone', ckp_name))
    # model.visual.load_state_dict(ckp['vision_encoder_state_dict'])

    if 'prompter' in (args.cls_prompt, args.atk_prompt):
        prompter_path = os.path.join(cfg.OUTPUT_DIR, 'prompt_learner/')
    
        assert os.path.isdir(prompter_path)
        if args.best_checkpoint:
            prompter_path += 'best.pth.tar'
        else:
            ckp = [fname for fname in os.listdir(prompter_path) if 'model.pth.tar' in fname][0]
            prompter_path += ckp
            
    classify_prompt = prompter_path if args.cls_prompt == 'prompter' else args.cls_prompt
    attack_prompt = prompter_path if args.atk_prompt == 'prompter' else args.atk_prompt
   
    if args.linear_probe:
        from adv_lp import LinearProbe
        model = LinearProbe(model, 512, num_classes, False)
        ckp = torch.load(os.path.join(cfg.OUTPUT_DIR, 'linear_probe/linear.pth.tar'))
        model.linear.load_state_dict(ckp)
    else:
        model = CustomCLIP(model,
                           classes,
                           cls_prompt=classify_prompt,
                           atk_prompt=attack_prompt,
                           cfg=cfg)
    model = model.cuda()
    prompts = model._get_prompts()
    model.eval()


    if args.cls_prompt == 'prompter':
        prompt_learner_state = torch.load(classify_prompt, map_location='cpu')["state_dict"]
        ctx = prompt_learner_state["ctx"]
        ctx = ctx.float()
        print(f"Size of context: {ctx.shape}")

        tokenizer = SimpleTokenizer()
        clip_model = load_clip_to_cpu()
        token_embedding = clip_model.token_embedding.weight
        print(f"Size of token embedding: {token_embedding.shape}")

        topk = args.topk
        if ctx.dim() == 2:
            # Generic context
            distance = torch.cdist(ctx, token_embedding)
            # print(f"Size of distance matrix: {distance.shape}")
            sorted_idxs = torch.argsort(distance, dim=1)
            sorted_idxs = sorted_idxs[:, :topk]
            raw_words = []
            for m, idxs in enumerate(sorted_idxs):
                words = [tokenizer.decoder[idx.item()].replace('</w>', '') for idx in idxs]
                # print(f"Context {m+1}: {' '.join(words)}")
                raw_words.extend(words)
            raw_phrase = ' '.join(raw_words)
            class_raw_titles = [f"{raw_phrase} {classes[class_idx]}." for class_idx in range(num_classes)]
        elif ctx.dim() == 3:
            # Class-specific context
            print("Processing class-specific context...")
            n_classes, n_ctx, dim = ctx.shape
            # print(f"Number of classes: {n_classes}, Context tokens per class: {n_ctx}, Dimension: {dim}")

            class_raw_words = []
            for class_idx, class_ctx in enumerate(ctx):
                # print(f"\nClass {class_idx + 1}:")
                distance = torch.cdist(class_ctx, token_embedding)
                # print(f"Size of distance matrix: {distance.shape}")

                sorted_idxs = torch.argsort(distance, dim=1)[:, :topk]
                words_per_class = []
                for m, idxs in enumerate(sorted_idxs):
                    words = [tokenizer.decoder[idx.item()].replace('</w>', '') for idx in idxs]
                    # print(f"  Context token {m+1}: {' '.join(words)}")
                    words_per_class.append(words[0])
                sentence = ' '.join(words_per_class)
                # print(f"Generated sentence for Class {class_idx + 1}: {sentence} class")
                class_raw_words.append(sentence)
            class_raw_titles = [f"{class_raw_words[class_idx]} {classes[class_idx]}" for class_idx in range(num_classes)]
        else:
            raise ValueError("Unsupported context dimension.")
    else:
        class_raw_titles = [args.cls_prompt.format(classes[class_idx]) for class_idx in range(num_classes)]
    
    meters = Dict()
    meters.acc = AverageMeter('Clean Acc@1', ':6.2f')
    meters.rob = AverageMeter('Robust Acc@1', ':6.2f')
    
    progress = ProgressMeter(
        len(loader),
        [meters.acc, meters.rob],
        prefix=cfg.DATASET.NAME)

    eps = cfg.AT.EPS
    alpha = eps / 4.0
    steps = 100

    if args.attack == 'aa':
        attack = AutoAttack(model,
                            norm='Linf',
                            eps=eps,
                            version='standard',
                            verbose=False)
    elif args.attack == 'pgd':
        attack = PGD(model, eps=eps, alpha=alpha, steps=steps)
    elif args.attack == 'tpgd':
        attack = TPGD(model, eps=eps, alpha=alpha, steps=steps)
    if args.save_img:
        clean_dir = os.path.join(args.save_path, 'clean_test')
        adv_dir = os.path.join(args.save_path, 'adv_test')
        os.makedirs(clean_dir, exist_ok=True)
        os.makedirs(adv_dir, exist_ok=True)
        print(f'Sucessfully create {clean_dir} for clean image directory!')
        print(f'Sucessfully create {adv_dir} for adversarial image directory!')
        
        all_logits_clean = []
        all_images_clean = []
        all_logits_adv = []
        all_images_adv = []
        all_labels = []


    for i, data in enumerate(loader, start=1):
        try:
            # few-shot data loader from Dassl
            imgs, tgts = data['img'], data['label']
        except:
            imgs, tgts = data[:2]
        imgs, tgts = imgs.cuda(), tgts.cuda()
        if args.save_img:
            all_labels.append(tgts.cpu())
        bs = imgs.size(0)

        with torch.no_grad():
            output = model(imgs)    
            if args.save_img:
                all_logits_clean.append(output.cpu())
                all_images_clean.append(imgs.cpu())

        acc = accuracy(output, tgts)
        meters.acc.update(acc[0].item(), bs)
        imgs.requires_grad_()
        model.mode = 'attack'
        if args.attack == 'aa':
            adv = attack.run_standard_evaluation(imgs, tgts, bs=bs)
        elif args.attack in ['pgd', 'tpgd']:
            adv = attack(imgs, tgts)
        else:
            adv, _ = pgd(imgs, tgts, model, CWLoss, eps, alpha, steps)
            
        model.mode = 'classification'

        # Calculate features
        with torch.no_grad():
            output = model(adv)
            if args.save_img:
                all_logits_adv.append(output.cpu())
                all_images_adv.append(adv.cpu())
        
        rob = accuracy(output, tgts)
        meters.rob.update(rob[0].item(), bs)

        if i == 1 or i % 10 == 0 or i == len(loader):
            progress.display(i)

    if args.save_img:
        all_logits_clean = torch.cat(all_logits_clean, dim=0)
        # print(f'all_logits_clean: {all_logits_clean.shape}')
        all_images_clean = torch.cat(all_images_clean, dim=0)
        # print(f'all_images_clean: {all_images_clean.shape}')
        all_logits_adv = torch.cat(all_logits_adv, dim=0)
        # print(f'all_logits_adv: {all_logits_adv.shape}')
        all_images_adv = torch.cat(all_images_adv, dim=0)
        # print(f'all_images_adv: {all_images_adv.shape}')
        all_labels = torch.cat(all_labels, dim=0)
        # print(f'all_labels: {all_labels.shape}')
        torch.manual_seed(args.seed)
        
        for class_idx in range(num_classes):    
            indices_for_class = (all_labels == class_idx).nonzero(as_tuple=False).squeeze()
            if indices_for_class.numel() == 0:
                print(f"No images found for class {classes[class_idx]}")
                continue
            
            logits_for_class_clean = all_logits_clean[indices_for_class]
            images_for_class_clean = all_images_clean[indices_for_class]
            logits_for_class_adv = all_logits_adv[indices_for_class]
            images_for_class_adv = all_images_adv[indices_for_class]

            # Select random images
            k = min(args.num_imgs, logits_for_class_clean.size(0))
            if k == 0:
                print(f"No images available for class {classes[class_idx]}. Skipping.")
                continue
            random_indices = torch.randperm(logits_for_class_clean.size(0))[:k]
            print(f"Selected {k} random clean images for class {classes[class_idx]}: {random_indices}")
            
            selected_logits_clean = logits_for_class_clean[random_indices]
            selected_images_clean = images_for_class_clean[random_indices]
            selected_logits_adv = logits_for_class_adv[random_indices]
            selected_images_adv = images_for_class_adv[random_indices]
            
            correct_clean_preds = (selected_logits_clean.argmax(dim=1) == class_idx).sum().item()
            incorrect_clean_preds = k - correct_clean_preds
            print(f"Correct predictions for clean images: {correct_clean_preds}/{k}")
            print(f"Incorrect predictions for clean images: {incorrect_clean_preds}/{k}")
            
            # Plot and save clean images
            fig, axes = plt.subplots(2, 5, figsize=(15, 6))
            fig.suptitle(class_raw_titles[class_idx], fontsize=16)
            for j, ax in enumerate(axes.flat):
                if j < len(selected_images_clean):
                    img = np.transpose(selected_images_clean[j], (1, 2, 0))
                    ax.imshow(img)
                    ax.axis('off')
                    predicted_class = classes[selected_logits_clean.argmax(dim=1)[j].item()]
                    ax.set_title(f"True class {classes[class_idx]}\nPredicted class {predicted_class}")
                else:
                    ax.axis('off')
            plt.savefig(os.path.join(clean_dir, f'class_{classes[class_idx]}_clean.png'))
            plt.close(fig)

            print(f"Selected {k} random adversarial images for class {classes[class_idx]}: {random_indices}")
            correct_adv_preds = (selected_logits_adv.argmax(dim=1) == class_idx).sum().item()
            incorrect_adv_preds = k - correct_adv_preds
            print(f"Correct predictions for adversarial images: {correct_adv_preds}/{k}")
            print(f"Incorrect predictions for adversarial images: {incorrect_adv_preds}/{k}")

            fig, axes = plt.subplots(2, 5, figsize=(15, 6))
            fig.suptitle(class_raw_titles[class_idx], fontsize=16)
            for j, ax in enumerate(axes.flat):
                if j < len(selected_images_adv):
                    img = np.transpose(selected_images_adv[j], (1, 2, 0))
                    ax.imshow(img)
                    ax.axis('off')
                    predicted_class = classes[selected_logits_adv.argmax(dim=1)[j].item()]
                    ax.set_title(f"True class {classes[class_idx]}\nPredicted class {predicted_class}")
                else:
                    ax.axis('off')
            plt.savefig(os.path.join(adv_dir, f'class_{classes[class_idx]}_adv.png'))
            plt.close(fig)

    # save result
    if os.path.isfile(save_path):
        with open(save_path, 'r') as f:
            result = Dict(yaml.safe_load(f))
    else:
        result = Dict()
        
    _result = result if args.dataset is None or args.dataset==train_dataset else result[args.dataset]
    tune = 'linear_probe' if args.linear_probe else args.cls_prompt
    _result[tune].clean = meters.acc.avg
    _result[tune][args.attack] = meters.rob.avg

    with open(save_path, 'w+') as f:
        yaml.dump(result.to_dict(), f)
    
    print(f'result saved at: {save_path}')
