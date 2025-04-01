import os
import torch
from warnings import warn
from yacs.config import CfgNode
import yaml
import argparse
from tqdm import tqdm
import sys

from clip.simple_tokenizer import SimpleTokenizer
from clip import clip
import matplotlib.pyplot as plt
import numpy as np

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
    CW loss (Margin loss).
    """
    num_classes = output.shape[-1]
    target = target.data
    target_onehot = torch.zeros(target.size() + (num_classes,))
    target_onehot = target_onehot.cuda()
    target_onehot.scatter_(1, target.unsqueeze(1), 1.0)
    target_var = Variable(target_onehot, requires_grad=False)
    real = (target_var * output).sum(1)
    other = ((1.0 - target_var) * output - target_var * 10000.0).max(1)[0]
    loss = -torch.clamp(real - other + confidence, min=0.0)
    loss = torch.sum(loss)
    return loss


def input_grad(imgs, targets, model, criterion):
    output = model(imgs)
    loss = criterion(output, targets)
    ig = grad(loss, imgs)[0]
    return ig


def perturb(imgs, targets, model, criterion, eps, eps_step, pert=None, ig=None):
    adv = imgs.requires_grad_(True) if pert is None else torch.clamp(imgs + pert, 0, 1).requires_grad_(True)
    ig = input_grad(adv, targets, model, criterion) if ig is None else ig
    if pert is None:
        pert = eps_step * torch.sign(ig)
    else:
        pert += eps_step * torch.sign(ig)
    pert.clamp_(-eps, eps)
    adv = torch.clamp(imgs + pert, 0, 1)
    pert = adv - imgs
    return adv.detach(), pert.detach()


def pgd(imgs, targets, model, criterion, eps, eps_step, max_iter, pert=None, ig=None):
    for i in range(max_iter):
        adv, pert = perturb(imgs, targets, model, criterion, eps, eps_step, pert, ig)
        ig = None
    return adv, pert


parser = argparse.ArgumentParser()
parser.add_argument('experiment')
parser.add_argument('-cp', '--cls-prompt', default='a photo of a {}')
parser.add_argument('-ap', '--atk-prompt', default='a photo of a {}')
parser.add_argument('--best-checkpoint', action='store_true')

parser.add_argument('--attack', default='pgd')
parser.add_argument('--dataset', default=None)
parser.add_argument('-lp', '--linear-probe', action='store_true')

if __name__ == '__main__':
    args = parser.parse_args()
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

        result = result if args.dataset is None or args.dataset == train_dataset else result[args.dataset]
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
        new_model = LinearProbe(model, 512, num_classes, False)
        ckp = torch.load(os.path.join(cfg.OUTPUT_DIR, 'linear_probe/linear.pth.tar'))
        new_model.linear.load_state_dict(ckp)
    else:
        new_model = CustomCLIP(model,
                               classes,
                               cls_prompt=classify_prompt,
                               atk_prompt=attack_prompt,
                               cfg=cfg)

    # Prepare text features for classification and attack
    cls_tfeatures = new_model._prompt_text_features(classify_prompt).cuda()
    if attack_prompt is None or classify_prompt == attack_prompt:
        atk_tfeatures = cls_tfeatures
    else:
        atk_tfeatures = new_model._prompt_text_features(attack_prompt).cuda()
    logit_scale = model.logit_scale.exp()
    new_model = new_model.cuda()
    new_model.eval()

    #BEGIN NEW CODE
    # Function to load CLIP model to CPU
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

    # Load the prompt learner and extract raw words
    if args.cls_prompt == 'prompter':
        prompt_learner_state = torch.load(classify_prompt, map_location='cpu')["state_dict"]
        ctx = prompt_learner_state["ctx"]
        ctx = ctx.float()
        print(f"Size of context: {ctx.shape}")

        # Load tokenizer and token embeddings
        tokenizer = SimpleTokenizer()
        clip_model = load_clip_to_cpu()
        token_embedding = clip_model.token_embedding.weight
        print(f"Size of token embedding: {token_embedding.shape}")

        topk = 1  # Number of top words to extract

        if ctx.dim() == 2:
            # Generic context
            distance = torch.cdist(ctx, token_embedding)
            print(f"Size of distance matrix: {distance.shape}")
            sorted_idxs = torch.argsort(distance, dim=1)
            sorted_idxs = sorted_idxs[:, :topk]
            raw_words = []
            for m, idxs in enumerate(sorted_idxs):
                words = [tokenizer.decoder[idx.item()].replace('</w>', '') for idx in idxs]
                print(f"Context {m+1}: {' '.join(words)}")
                raw_words.extend(words)
            raw_phrase = ' '.join(raw_words)
            class_raw_titles = [f"{raw_phrase} {classes[class_idx]}." for class_idx in range(num_classes)]
        elif ctx.dim() == 3:
            # Class-specific context
            print("Processing class-specific context...")
            n_classes, n_ctx, dim = ctx.shape
            print(f"Number of classes: {n_classes}, Context tokens per class: {n_ctx}, Dimension: {dim}")

            class_raw_words = []
            for class_idx, class_ctx in enumerate(ctx):  # class_ctx: [n_ctx, dim]
                print(f"\nClass {class_idx + 1}:")
                distance = torch.cdist(class_ctx, token_embedding)  # [n_ctx, vocab_size]
                print(f"Size of distance matrix: {distance.shape}")

                sorted_idxs = torch.argsort(distance, dim=1)[:, :topk]
                words_per_class = []
                for m, idxs in enumerate(sorted_idxs):
                    words = [tokenizer.decoder[idx.item()].replace('</w>', '') for idx in idxs]
                    print(f"  Context token {m+1}: {' '.join(words)}")
                    words_per_class.append(words[0])  # Take top-1 word
                sentence = ' '.join(words_per_class)
                print(f"Generated sentence for Class {class_idx + 1}: {sentence} class")
                class_raw_words.append(sentence)
            class_raw_titles = [f"{class_raw_words[class_idx]} {classes[class_idx]}" for class_idx in range(num_classes)]
        else:
            raise ValueError("Unsupported context dimension.")
    else:
	    # If not using prompter, generate titles using the provided prompt format
        class_raw_titles = [args.cls_prompt.format(classes[class_idx]) for class_idx in range(num_classes)]
    # Proceed with the rest of the code
    eps = cfg.AT.EPS
    alpha = eps / 4.0
    steps = 100

    if args.attack == 'aa':
        attack = AutoAttack(new_model,
                            norm='Linf',
                            eps=eps,
                            version='standard',
                            verbose=False)
    elif args.attack == 'pgd':
        attack = PGD(new_model, eps=eps, alpha=alpha, steps=steps)
    elif args.attack == 'tpgd':
        attack = TPGD(new_model, eps=eps, alpha=alpha, steps=steps)

    import heapq
    base_dir = '/kaggle/working/'
    clean_dir = os.path.join(base_dir, 'clean_test')
    adv_dir = os.path.join(base_dir, 'adv_test_wo_lw')
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(adv_dir, exist_ok=True)

    # Initialize data structures to store top 10 images per class
    top_images_clean = {class_idx: [] for class_idx in range(num_classes)}
    top_images_adv = {class_idx: [] for class_idx in range(num_classes)}
    mu = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    normalize = ImageNormalizer(mu, std).cuda()
    # Iterate over batches
    for i, data in enumerate(loader, start=1):
        print(f'Processing batch {i}...')
        try:
            imgs, tgts = data['img'], data['label']
        except:
            imgs, tgts = data[:2]
        imgs, tgts = imgs.cuda(), tgts.cuda()
        bs = imgs.size(0)

        # Generate adversarial images
        new_model.mode = 'attack'
        if args.attack == 'aa':
            adv = attack.run_standard_evaluation(imgs, tgts, bs=bs)
        elif args.attack in ['pgd', 'tpgd']:
            adv = attack(imgs, tgts)
        else:
            adv, _ = pgd(imgs, tgts, new_model, CWLoss, eps, alpha, steps)
        new_model.mode = 'classification'
        with torch.no_grad():
            # Encode image features
            image_features_clean = model.encode_image(normalize(imgs))
            image_features_clean = image_features_clean / image_features_clean.norm(dim=-1, keepdim=True)

            image_features_adv = model.encode_image(normalize(adv))
            image_features_adv = image_features_adv / image_features_adv.norm(dim=-1, keepdim=True)

            # Move images to CPU to save GPU memory
            imgs_cpu = imgs.detach().cpu().numpy()
            adv_cpu = adv.detach().cpu().numpy()

        # Iterate over classes
        for class_idx in range(num_classes):
            text_feature = cls_tfeatures[class_idx].cuda()

            # Compute cosine similarities with clean images
            similarities_clean = (logit_scale * image_features_clean @ text_feature).detach().cpu().numpy()

            # Update top 10 clean images per class
            for sim, img in zip(similarities_clean, imgs_cpu):
                heapq.heappush(top_images_clean[class_idx], (sim, img))
                if len(top_images_clean[class_idx]) > 10:
                    heapq.heappop(top_images_clean[class_idx])

            # Compute cosine similarities with adversarial images
            similarities_adv = (logit_scale * image_features_adv @ text_feature).detach().cpu().numpy()

            # Update top 10 adversarial images per class
            for sim, img in zip(similarities_adv, adv_cpu):
                heapq.heappush(top_images_adv[class_idx], (sim, img))
                if len(top_images_adv[class_idx]) > 10:
                    heapq.heappop(top_images_adv[class_idx])

    print('Processing complete!')

    # Save results
    for class_idx in range(num_classes):
        print(f'Class {classes[class_idx]}:')

        # Use a single title for the entire set of images for the class
        title = class_raw_titles[class_idx]

        # Sort clean images by similarity
        top_images_clean[class_idx].sort(key=lambda x: x[0], reverse=True)
        imgs_clean = [img for sim, img in top_images_clean[class_idx]]

        # Display and save top 10 clean images
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        for j, ax in enumerate(axes.flat):
            if j < len(imgs_clean):
                img = np.transpose(imgs_clean[j], (1, 2, 0))
                ax.imshow(img)
                ax.axis('off')
            else:
                ax.axis('off')
        # Set the title for the entire figure (not individual images)
        fig.suptitle(title, fontsize=16)
        plt.savefig(os.path.join(clean_dir, f'top_images_class_{classes[class_idx]}_clean.png'))
        plt.close(fig)
        print(f"Saved {len(imgs_clean)} clean images for class {classes[class_idx]} to {clean_dir}")

        # Sort adversarial images by similarity
        top_images_adv[class_idx].sort(key=lambda x: x[0], reverse=True)
        imgs_adv = [img for sim, img in top_images_adv[class_idx]]

        # Display and save top 10 adversarial images
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        for j, ax in enumerate(axes.flat):
            if j < len(imgs_adv):
                img = np.transpose(imgs_adv[j], (1, 2, 0))
                ax.imshow(img)
                ax.axis('off')
            else:
                ax.axis('off')
        # Set the title for the entire figure
        fig.suptitle(title, fontsize=16)
        plt.savefig(os.path.join(adv_dir, f'top_images_class_{classes[class_idx]}_adv.png'))
        plt.close(fig)
        print(f"Saved {len(imgs_adv)} adversarial images for class {classes[class_idx]} to {adv_dir}")
    #END NEW CODE
