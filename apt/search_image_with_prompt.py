import os
import torch
from warnings import warn
from yacs.config import CfgNode
import yaml
import argparse
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np

from statistics import mean

from torchvision import transforms
from torchvision.datasets import *

from torch.autograd import grad, Variable

from addict import Dict

from dassl.data import DataManager

import datasets.oxford_pets
import datasets.oxford_flowers
import datasets.fgvc_aircraft
import datasets.dtd
import datasets.eurosat
import datasets.stanford_cars
import datasets.food101
import datasets.sun397
import datasets.caltech101
import datasets.ucf101
import datasets.imagenet


from torchattacks import PGD, TPGD
from autoattack import AutoAttack

from utils import *


def CWLoss(output, target, confidence=0):
    """
    CW loss (Marging loss).
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

PROMPT_PRESETS = {
    "default": "a photo of a {}",
    "special": "a special picture of a {}",
    "drawing": "a drawing of a {}",
    "adversarial": "an adversarial example of a {}"
}

parser = argparse.ArgumentParser()
parser.add_argument('experiment')
parser.add_argument('-cp','--cls-prompt', default='default')
parser.add_argument('-ap','--atk-prompt', default=None)
parser.add_argument('--best-checkpoint', action='store_true')

parser.add_argument('--attack', default='pgd')
parser.add_argument('--dataset', default=None)
parser.add_argument('-lp', '--linear-probe', action='store_true')
def get_prompt(preset_or_custom, default="default"):
    if preset_or_custom in PROMPT_PRESETS:
        return PROMPT_PRESETS[preset_or_custom]
    return preset_or_custom

# Lấy prompt từ preset hoặc custom


if __name__ == '__main__':
    args = parser.parse_args()
    cls_prompt = get_prompt(args.cls_prompt)
    atk_prompt = get_prompt(args.atk_prompt) if args.atk_prompt else None
    cfg = CfgNode()
    cfg.set_new_allowed(True)
    cfg_path = os.path.join(args.experiment, 'cfg.yaml')
    cfg.merge_from_file(cfg_path)

    train_dataset = cfg.DATASET.NAME
    
    if args.dataset:
        if args.dataset in ['ImageNetR', 'ImageNetA', 'ON']:
            cfg.DATASET.NAME = 'ImageNet'
        else:
            cfg.DATASET.NAME = args.dataset
        save_path = os.path.join(cfg.OUTPUT_DIR, 'dist_shift.yaml')
    else:
        save_path = os.path.join(cfg.OUTPUT_DIR, 'evaluation.yaml')
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
    ckp_name = 'vitb32' if cfg.MODEL.BACKBONE.NAME == 'ViT-B/32' else 'rn50'
    eps = int(cfg.AT.EPS * 255)
    ckp_name += f'_eps{eps}.pth.tar'
    ckp = torch.load(os.path.join('backbone', ckp_name))
    model.visual.load_state_dict(ckp['vision_encoder_state_dict'])

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
    model.eval()

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
    #BEGIN NEW CODE  
    base_dir = '/kaggle/working'
    clean_dir = os.path.join(base_dir, 'clean_test')
    adv_dir = os.path.join(base_dir, 'adv_test')
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(adv_dir, exist_ok=True)

    all_images_clean = []
    all_images_adv = []

    # Duyệt qua từng batch và lưu lại logits và ảnh cho clean và adv
    for i, data in enumerate(loader, start=1):
        try:
            imgs, tgts = data['img'], data['label']
        except:
            imgs, tgts = data[:2]
        imgs, tgts = imgs.cuda(), tgts.cuda()
        bs = imgs.size(0)

        all_images_clean.append(imgs.cpu())

        # Áp dụng tấn công để tạo ảnh adversarial
        model.mode = 'attack'
        if args.attack == 'aa':
            adv = attack.run_standard_evaluation(imgs, tgts, bs=bs)
        elif args.attack in ['pgd', 'tpgd']:
            adv = attack(imgs, tgts)
        else:
            adv, _ = pgd(imgs, tgts, model, CWLoss, eps, alpha, steps)
        all_images_adv.append(adv.cpu())
    
    # Kết hợp tất cả các ảnh thành tensor
    all_images_clean = torch.cat(all_images_clean, dim=0)
    all_images_adv = torch.cat(all_images_adv, dim=0)

    # Chuẩn bị text features cho classification và attack
    cls_tfeatures = model._prompt_text_features(classify_prompt).cuda()  # Mã hóa text features
    if attack_prompt is None or classify_prompt == attack_prompt:
        atk_tfeatures = cls_tfeatures
    else:
        atk_tfeatures = model._prompt_text_features(attack_prompt).cuda()

    # Chuẩn hóa ảnh và mã hóa image features
    normalize = ImageNormalizer(mu, std).cuda()
    image_features_clean = model.encode_image(normalize(all_images_clean))
    image_features_clean = image_features_clean / image_features_clean.norm(dim=-1, keepdim=True)
    
    image_features_adv = model.encode_image(normalize(all_images_adv))
    image_features_adv = image_features_adv / image_features_adv.norm(dim=-1, keepdim=True)

    # Duyệt qua từng lớp
    for class_idx in range(num_classes):
        # Lấy text feature của lớp hiện tại
        text_feature = cls_tfeatures[class_idx].unsqueeze(0)  # Lấy vector nhúng của lớp hiện tại (1xD)

        # Tính độ tương đồng cosine với tất cả ảnh sạch
        similarities_clean = (image_features_clean @ text_feature.T).squeeze(1)
        top_values_clean, top_indices_clean = torch.topk(similarities_clean, k=10)  # Lấy top 10

        # Lấy các ảnh top 10 cho clean
        top_images_clean = [all_images_clean[i].cpu().numpy() for i in top_indices_clean]

        # Hiển thị và lưu kết quả top 10 ảnh sạch
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        for j, ax in enumerate(axes.flat):
            if j < len(top_images_clean):
                img = np.transpose(top_images_clean[j], (1, 2, 0))
                ax.imshow(img)
                ax.axis('off')
                ax.set_title(f"Class {classes[class_idx]}")
            else:
                ax.axis('off')
        plt.savefig(os.path.join(clean_dir, f'top_images_class_{classes[class_idx]}_clean.png'))
        print(f"Saved top {len(top_images_clean)} clean images for class {classes[class_idx]} to {clean_dir}")

        # Tính độ tương đồng cosine với tất cả ảnh đối nghịch
        similarities_adv = (image_features_adv @ text_feature.T).squeeze(1)
        top_values_adv, top_indices_adv = torch.topk(similarities_adv, k=10)  # Lấy top 10

        # Lấy các ảnh top 10 cho adversarial
        top_images_adv = [all_images_adv[i].cpu().numpy() for i in top_indices_adv]

        # Hiển thị và lưu kết quả top 10 ảnh đối nghịch
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        for j, ax in enumerate(axes.flat):
            if j < len(top_images_adv):
                img = np.transpose(top_images_adv[j], (1, 2, 0))
                ax.imshow(img)
                ax.axis('off')
                ax.set_title(f"Class {classes[class_idx]}")
            else:
                ax.axis('off')
        plt.savefig(os.path.join(adv_dir, f'top_images_class_{classes[class_idx]}_adv.png'))
        print(f"Saved top {len(top_images_adv)} adversarial images for class {classes[class_idx]} to {adv_dir}")

    #END NEW CODE
   