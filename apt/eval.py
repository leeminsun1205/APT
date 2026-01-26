import multiprocessing as mp
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import os
import torch
from yacs.config import CfgNode
import yaml
import json
import argparse
import clip
import numpy as np
from transformers import AutoTokenizer, AutoProcessor, AlignModel
from torch.autograd import grad, Variable
from addict import Dict
from torchvision import transforms
from torch.utils.data import DataLoader
from dassl.data import DataManager
from torch.utils.data import SequentialSampler
from lavis.models import load_model_and_preprocess
from torchvision.transforms import ToPILImage


from datasets import (
    oxford_pets, oxford_flowers, fgvc_aircraft, dtd, eurosat, 
    stanford_cars, food101, sun397, caltech101, ucf101, cifar,
    tiny_imagenet
)
from robustbench.data import load_cifar10c, load_cifar100c
from torch.utils.data import TensorDataset


import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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


parser = argparse.ArgumentParser()
parser.add_argument('experiment', type=str, help="Name or type of experiment to run.")
parser.add_argument('-cp','--cls-prompt', type=str, default='a photo of a {}')
parser.add_argument('-ap','--atk-prompt', type=str, default=None)
parser.add_argument('--best-checkpoint', action='store_true')
parser.add_argument('--rob', action='store_true')
parser.add_argument('--model', default='CLIP', choices=['ALIGN', 'BLIP', 'CLIP'])
parser.add_argument('--save-path', type=str, default = './',
                    help="Specific path to save images. Default is ./")
parser.add_argument('--attack', default='pgd')
parser.add_argument('--dataset', default=None)
parser.add_argument('-lp', '--linear-probe', action='store_true')
parser.add_argument('-at', '--pre_AT', action='store_true')
parser.add_argument('-bs', '--batch-size', type=int, default=100)
parser.add_argument('--subset', type=int, default=None, help="Test on a subset of the dataset (first N samples)")
parser.add_argument('-atk_e', '--atk_eps', type=int, default = None)

if __name__ == '__main__':

    args = parser.parse_args()

    # Normalize dataset name to canonical case usually expected by the script
    if args.dataset:
        canonical_names = {
            'cifar10c': 'Cifar10C',
            'cifar100c': 'Cifar100C',
            'imagenetr': 'ImageNetR',
            'imageneta': 'ImageNetA',
            'imagenetv2': 'ImageNetV2',
            'on': 'ON',
            'cifar10': 'CIFAR10',
            'cifar100': 'CIFAR100'
        }
        args.dataset = canonical_names.get(args.dataset.lower(), args.dataset)

    cfg = CfgNode()
    cfg.set_new_allowed(True)
    cfg_path = os.path.join(args.experiment, 'cfg.yaml')
    cfg.merge_from_file(cfg_path)

    train_dataset = cfg.DATASET.NAME
    if args.save_path:
        save_output = args.save_path
    else:
        save_output = cfg.OUTPUT_DIR
    os.makedirs(save_output, exist_ok=True)
    if args.dataset:
        if args.dataset in ['ImageNetR', 'ImageNetA', 'ON']:
            cfg.DATASET.NAME = 'ImageNet'
        elif args.dataset == 'Cifar10C':
            cfg.DATASET.NAME = 'CIFAR10'
        elif args.dataset in ['Cifar100C']:
            cfg.DATASET.NAME = 'CIFAR100'
        else:
            cfg.DATASET.NAME = args.dataset
        save_path = os.path.join(save_output, f'{args.model}_{args.dataset}.yaml')
    else:
        save_path = os.path.join(save_output, 'evaluation.yaml')
    if os.path.isfile(save_path):
        with open(save_path, 'r') as f:
            result = Dict(yaml.safe_load(f))

        result = result if args.dataset is None or args.dataset==train_dataset else result[args.dataset]
        tune = 'linear_probe' if args.linear_probe else args.cls_prompt
        if result[tune][args.attack] != {}:
            print(f'eval result already exists at: {save_path}')
            exit()
    num_classes = None
    classes = None
    loader = None
    

    cfg.DATALOADER.TEST.BATCH_SIZE = args.batch_size
    cfg.DATALOADER.NUM_WORKERS = 4

    if cfg.DATASET.NUM_LABELED <= 0:
        cfg.DATASET.NUM_LABELED = 10  # Default value to bypass assertion

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

    elif args.dataset in ['Cifar10C', 'Cifar100C', 'CIFAR10C', 'CIFAR100C']:
        # Default to severity 5 as standard benchmark, or make it configurable if needed
        severity = 5
        n_examples = 10000 # Full test set size for CIFAR
        data_dir = './data' # Or use cfg.DATASET.ROOT if appropriate, but robustbench usually handles downloads

        if args.dataset == 'Cifar10C':
             print(f"Loading CIFAR-10-C (Severity {severity})...")
             x_test, y_test = load_cifar10c(n_examples=n_examples, severity=severity, data_dir=data_dir)
             num_classes = 10
             classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


        elif args.dataset == 'Cifar100C':
             print(f"Loading CIFAR-100-C (Severity {severity})...")
             x_test, y_test = load_cifar100c(n_examples=n_examples, severity=severity, data_dir=data_dir)
             num_classes = 100
             from torchvision.datasets import CIFAR100
             dummy_ds = CIFAR100(root='./data', download=True, train=False)
             classes = dummy_ds.classes

        dataset = TensorDataset(x_test, y_test)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    if args.subset:
        if args.subset < len(loader.dataset):
            print(f"*** Subsetting test set to first {args.subset} samples ***")
            subset_indices = list(range(args.subset))
            subset_dataset = torch.utils.data.Subset(loader.dataset, subset_indices)
            loader = DataLoader(subset_dataset, 
                                batch_size=args.batch_size, 
                                shuffle=False, 
                                num_workers=4, 
                                pin_memory=True)
        else:
            print(f"Warning: Subset size {args.subset} >= dataset size {len(loader.dataset)}. Using full dataset.")

        
    model, processor, tokenizer = None, None, None
    if args.model == 'CLIP':
        print('model: CLIP')
        model, _ = clip.load(cfg.MODEL.BACKBONE.NAME, device='cpu')
    elif args.model == 'ALIGN':
        print('model: ALIGN')
        model = AlignModel.from_pretrained("kakaobrain/align-base")
        processor = AutoProcessor.from_pretrained("kakaobrain/align-base")
        tokenizer = AutoTokenizer.from_pretrained("kakaobrain/align-base")
    elif args.model == 'BLIP':
        print('model: BLIP')
        model, _, _ = load_model_and_preprocess("blip_feature_extractor", model_type="base", is_eval=True, device='cuda')
    else:
        raise ValueError(f'Unknown model: {args.model}')
    
    if args.pre_AT:
        ckp_name = 'vitb32' if cfg.MODEL.BACKBONE.NAME == 'ViT-B/32' else 'rn50'
        eps = int(cfg.AT.EPS * 255)
        ckp_name += f'_eps{eps}.pth.tar'

        ckp = torch.load(
            os.path.join('backbone', ckp_name),
            map_location='cpu',
        )

        state = ckp['vision_encoder_state_dict']

        # ===== Case 1: OpenAI CLIP (clip.load) =====
        if hasattr(model, 'visual'):
            print('[INFO] Loading AT weights into OpenAI CLIP visual encoder')
            missing, unexpected = model.visual.load_state_dict(state, strict=False)

        # ===== Case 2: HuggingFace-style CLIP =====
        elif hasattr(model, 'vision_model'):
            print('[INFO] Loading AT weights into HF CLIP vision_model')
            missing, unexpected = model.vision_model.load_state_dict(state, strict=False)

        else:
            raise RuntimeError('Unknown CLIP vision backbone structure')

        print('[INFO] AT backbone loaded')
        print('Missing keys:', missing)
        print('Unexpected keys:', unexpected)


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
        if args.model == 'ALIGN':
            model = CustomALIGN(model,
                            tokenizer,
                            classes,
                            cls_prompt=classify_prompt,
                            atk_prompt=attack_prompt,)
    
        elif args.model == 'BLIP':
            model = CustomBLIP(model,
                            classes,
                            cls_prompt=classify_prompt,
                            atk_prompt=attack_prompt,)
        elif args.model == 'CLIP':
            model = CustomCLIP(model,
                            classes,
                            cls_prompt=classify_prompt,
                            atk_prompt=attack_prompt,
                            cfg=cfg)
        
    model = model.cuda()
    model.eval()

    meters = Dict()
    meters.acc = AverageMeter('Clean Acc@1', ':6.2f')
    meters.rob = AverageMeter('Robust Acc@1', ':6.2f')

    # --- ADD RESUME LOGIC START ---
    progress_file_path = save_path + ".progress.json"
    start_batch = 1

    if os.path.exists(progress_file_path):
        print(f"*** Found progress file, reloading: {progress_file_path} ***")
        try:
            with open(progress_file_path, 'r') as f:
                progress_data = json.load(f)
            
            start_batch = progress_data['last_batch'] + 1
            meters.acc.sum = progress_data['acc_sum']
            meters.acc.count = progress_data['acc_count']
            meters.acc.avg = meters.acc.sum / meters.acc.count if meters.acc.count > 0 else 0
            meters.rob.sum = progress_data['rob_sum']
            meters.rob.count = progress_data['rob_count']
            meters.rob.avg = meters.rob.sum / meters.rob.count if meters.rob.count > 0 else 0
            print(f"*** Reload success! Resuming from batch {start_batch} ***")
        except Exception as e:
            print(f"Error loading progress file, starting from scratch. Error: {e}")
            start_batch = 1
            meters.acc.reset()
            meters.rob.reset()
    # --- ADD RESUME LOGIC END ---
    
    progress = ProgressMeter(
        len(loader),
        [meters.acc, meters.rob],
        prefix=cfg.DATASET.NAME)
    if args.atk_eps == None:
        eps = cfg.AT.EPS
        
    else:
        eps = args.atk_eps
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
        
    for i, data in enumerate(loader, start=1):
        if i < start_batch:
            continue
            
        try:
            imgs, tgts = data['img'], data['label']
        except:
            imgs, tgts = data[:2]
        imgs, tgts = imgs.cuda(), tgts.cuda()
        bs = imgs.size(0)
        image_inputs = imgs
        if args.model == 'ALIGN':
            imgs = [ToPILImage()(img.float()) for img in imgs]
            image_inputs = processor(images=imgs, return_tensors="pt")
            image_inputs = {k: v.cuda() for k, v in image_inputs.items()}
        with torch.no_grad():
            output = model(image_inputs)
        acc = accuracy(output, tgts)
        meters.acc.update(acc[0].item(), bs)
        if args.rob:
            model.mode = 'attack'

            if args.model == 'BLIP':
                pixel_values = imgs
            elif args.model == 'ALIGN':
                pixel_values = image_inputs["pixel_values"]
                pixel_values.requires_grad_()
            elif args.model == 'CLIP':
                pixel_values = imgs.requires_grad_()

            if args.attack == 'aa':
                advs = attack.run_standard_evaluation(pixel_values, tgts, bs=bs)
            elif args.attack in ['pgd', 'tpgd']:
                advs = attack(pixel_values, tgts)
            else:
                advs, _ = pgd(pixel_values, tgts, model, CWLoss, eps, alpha, steps)

            if args.model == 'ALIGN':
                advs = [ToPILImage()(adv.float()) for adv in advs]
                adv_inputs = processor(images=advs, return_tensors="pt")
                adv_inputs = {k: v.cuda() for k, v in adv_inputs.items()}
            else:
                adv_inputs = advs

            model.mode = 'classification'

            # Calculate features
            with torch.no_grad():
                output = model(adv_inputs)

            rob = accuracy(output, tgts)
            meters.rob.update(rob[0].item(), bs)

        if i == 1 or i % 10 == 0 or i == len(loader):
            progress.display(i)


            progress_data = {
                'last_batch': i,
                'acc_sum': meters.acc.sum,
                'acc_count': meters.acc.count,
                'rob_sum': meters.rob.sum,
                'rob_count': meters.rob.count
            }
            try:
                with open(progress_file_path, 'w') as f:
                    json.dump(progress_data, f)
            except Exception as e:
                print(f"Warning: Could not save progress file. {e}")
            
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

    # --- CLEANUP PROGRESS FILE ---
    if os.path.exists(progress_file_path):
        try:
            os.remove(progress_file_path)
            print(f"*** Evaluation complete. Deleted progress file: {progress_file_path} ***")
        except OSError as e:
            print(f"Warning: Could not delete progress file. {e}")