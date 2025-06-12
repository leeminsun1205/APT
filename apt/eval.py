import os
import torch
from yacs.config import CfgNode
import yaml
import argparse
from torchvision.datasets import *
from transformers import AutoTokenizer, AutoProcessor, AlignModel, Blip2Model
from torch.autograd import grad, Variable
from torchvision.datasets import CIFAR10
from addict import Dict
from torchvision import transforms
from torch.utils.data import DataLoader
from dassl.data import DataManager
from torch.utils.data import SequentialSampler
from lavis.models import load_model_and_preprocess

from datasets import (
    oxford_pets, oxford_flowers, fgvc_aircraft, dtd, eurosat, 
    stanford_cars, food101, sun397, caltech101, ucf101, imagenet
)

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

if __name__ == '__main__':

    args = parser.parse_args()
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
    if args.dataset == 'Cifar10':     
        num_classes = 10
        classes = [
            'airplanes',
            'cars',
            'birds',
            'cats',
            'deers',
            'dogs',
            'frogs',
            'horses',
            'ships',
            'trucks',
        ]
        if args.model == 'BLIP':
            _, processor = clip.load('ViT-B/32', device='cuda', jit=False)
        else:
            processor = transforms.Compose([
                transforms.ToTensor()
            ])
        testset = CIFAR10(root='./data', transform=processor, train=False, download=True)
        loader = DataLoader(testset,
                       batch_size=args.batch_size,
                       num_workers=4,
                       sampler=SequentialSampler(testset),)

    else:    
        cfg.DATALOADER.TEST.BATCH_SIZE = args.batch_size
        cfg.DATALOADER.TEST.NUM_WORKERS = 4
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
        ckp = torch.load(os.path.join('backbone', ckp_name))
        model.vision_model.load_state_dict(ckp['vision_encoder_state_dict'], strict=False)

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
        
    for i, data in enumerate(loader, start=1):
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