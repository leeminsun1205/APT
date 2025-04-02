import os
import torch
from yacs.config import CfgNode
import yaml
import argparse
from torchvision.datasets import *
from transformers import BlipProcessor, BlipModel
from PIL import Image
from torch.autograd import grad, Variable
from addict import Dict
from dassl.data import DataManager
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
#####################################################################################################
class CustomBLIP(torch.nn.Module):
    def __init__(self, processor, model, classnames, cls_prompt_template="a photo of a {}"):
        super().__init__()
        self.processor = processor
        self.model = model
        self.classnames = classnames
        self.cls_prompt_template = cls_prompt_template
        self.device = next(model.parameters()).device
        self.text_prompts = [self.cls_prompt_template.format(c) for c in self.classnames]
        text_inputs = self.processor(text=self.text_prompts, return_tensors="pt", padding=True)
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        with torch.no_grad():
            self.text_features = self.model.get_text_features(**text_inputs)

    def forward(self, image):
        pixel_values = image.to(self.device)
        image_features = self.model.get_image_features(pixel_values=pixel_values)
        logits_per_image = image_features @ self.text_features.t()
        return logits_per_image
#####################################################################################################
def input_grad(imgs, targets, model, criterion):
    imgs.requires_grad_(True)
    output = model(imgs)
    loss = criterion(output, targets)
    model.zero_grad()
    ig = grad(loss, imgs, retain_graph=False)[0]
    imgs.requires_grad_(False)
    return ig

def perturb(imgs, targets, model, criterion, eps, eps_step, pert=None, ig=None):
    imgs_clone = imgs.clone().detach()
    adv = imgs_clone if pert is None else torch.clamp(imgs_clone+pert, 0, 1)
    adv.requires_grad_(True)
    ig_calc = input_grad(adv, targets, model, criterion) if ig is None else ig
    adv.requires_grad_(False)
    if pert is None:
        pert = eps_step * torch.sign(ig_calc)
    else:
        pert += eps_step * torch.sign(ig_calc)
    pert = torch.clamp(pert, -eps, eps)
    adv_final = torch.clamp(imgs_clone + pert, 0, 1)
    final_pert = adv_final.detach() - imgs.detach()
    return adv_final.detach(), final_pert.detach()


def pgd(imgs, targets, model, criterion, eps, eps_step, max_iter, pert=None, ig=None):
    imgs_perturbed = imgs.clone().detach()
    current_pert = pert.clone().detach() if pert is not None else torch.zeros_like(imgs).detach()
    for i in range(max_iter):
        adv_for_grad = imgs_perturbed.requires_grad_(True)
        output = model(adv_for_grad)
        loss = criterion(output, targets)
        model.zero_grad()
        ig_step = grad(loss, adv_for_grad, retain_graph=False)[0]
        adv_for_grad.requires_grad_(False)
        current_pert += eps_step * torch.sign(ig_step.detach())
        current_pert = torch.clamp(current_pert, -eps, eps)
        imgs_perturbed = torch.clamp(imgs + current_pert, 0, 1).detach()
    final_pert = imgs_perturbed - imgs
    return imgs_perturbed, final_pert


parser = argparse.ArgumentParser()
parser.add_argument('experiment')
parser.add_argument('-cp','--cls-prompt', default='a photo of a {}')
parser.add_argument('-ap','--atk-prompt', default=None)
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
        save_path = os.path.join(cfg.OUTPUT_DIR, f'dist_shift_{args.dataset}.yaml')
    else:
        save_path = os.path.join(cfg.OUTPUT_DIR, 'evaluation.yaml')

    if os.path.isfile(save_path):
        with open(save_path, 'r') as f:
            result = Dict(yaml.safe_load(f))
        current_eval_dataset = args.dataset if args.dataset else train_dataset
        if current_eval_dataset not in result:
             result[current_eval_dataset] = Dict()
        tune_key = 'linear_probe' if args.linear_probe else 'zero_shot'
        if tune_key not in result[current_eval_dataset]:
            result[current_eval_dataset][tune_key] = Dict()
        attack_key = args.attack if args.attack else 'clean'
        if attack_key in result[current_eval_dataset][tune_key] and result[current_eval_dataset][tune_key][attack_key] is not None:
            print(f'Eval result for {current_eval_dataset}/{tune_key}/{attack_key} already exists at: {save_path}')
            # exit()
    else:
        result = Dict()

    dm = DataManager(cfg)
    classes = dm.dataset.classnames
    loader = dm.test_loader
    num_classes = dm.num_classes

    if args.dataset in ['ImageNetR', 'ImageNetA', 'ON', 'ImageNetV2'] or (train_dataset == 'ImageNet' and args.dataset is None and args.attack == 'aa'):
        try:
            from OODRB.imagenet import ImageNet # Assume this import works and handles different shifts
        except ImportError:
             print("Error: OODRB.imagenet not found. Please ensure it's installed or adjust the import.")
             exit()
        if args.dataset == 'ImageNetV2':
            shift = 'v2'
        elif args.dataset == 'ImageNetA':
            shift = 'A'
        elif args.dataset == 'ImageNetR':
            shift = 'R'
        elif args.dataset == 'ON':
            shift = 'ON'
        else:
             shift = 'val'
        num_classes = 1000
        temp_model_name = "Salesforce/blip-image-captioning-base"
        try:
            temp_processor = BlipProcessor.from_pretrained(temp_model_name)
        except Exception as e:
            print(f"Error loading BLIP processor {temp_model_name}: {e}")
            exit()
        image_transform = temp_processor.image_processor
        try:
            dataset = ImageNet(cfg.DATASET.ROOT,
                               shift=shift, # Pass shift explicitly if needed by ImageNet class
                               split='val', # Assuming 'val' split
                               transform=image_transform)
        except Exception as e:
             print(f"Error initializing ImageNet dataset: {e}")
             exit()

        if args.attack == 'aa' and len(dataset) > 5000:
            dataset = torch.utils.data.Subset(dataset, list(range(5000)))
        loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                                             shuffle=False,
                                             num_workers=cfg.DATALOADER.NUM_WORKERS,
                                             pin_memory=True)

    model_name = "Salesforce/blip-image-captioning-base"
    try:
        processor = BlipProcessor.from_pretrained(model_name)
        hf_blip_model = BlipModel.from_pretrained(model_name)
    except Exception as e:
        print(f"Error loading BLIP model/processor {model_name}: {e}")
        exit()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hf_blip_model.to(device)


    if 'prompter' in (args.cls_prompt, args.atk_prompt):
        prompter_path = os.path.join(cfg.OUTPUT_DIR, 'prompt_learner/')
        assert os.path.isdir(prompter_path), f"Prompter directory not found: {prompter_path}"
        if args.best_checkpoint:
            prompter_checkpoint_path = os.path.join(prompter_path, 'best.pth.tar')
        else:
            ckp_files = [fname for fname in os.listdir(prompter_path) if 'model.pth.tar' in fname]
            if not ckp_files:
                raise FileNotFoundError(f"No 'model.pth.tar' checkpoint found in {prompter_path}")
            prompter_checkpoint_path = os.path.join(prompter_path, ckp_files[0])

        if args.cls_prompt == 'prompter':
             print(f"WARNING: Loading actual prompter logic from {prompter_checkpoint_path} not implemented. Using path as placeholder.")
             classify_prompt = args.cls_prompt
        else:
             classify_prompt = args.cls_prompt
        attack_prompt = None
    else:
        classify_prompt = args.cls_prompt
        attack_prompt = args.atk_prompt


    if args.linear_probe:
        print("WARNING: Linear Probe path needs specific adaptation for Hugging Face BLIP features.")
        exit("Linear Probe adaptation required. Exiting.")
    else:
        print(classify_prompt)
        exit()
        model = CustomBLIP(processor,
                           hf_blip_model,
                           classes,
                           cls_prompt_template=classify_prompt)

    model.eval()

    meters = Dict()
    meters.acc = AverageMeter('Clean Acc@1', ':6.2f')
    meters.rob = AverageMeter('Robust Acc@1', ':6.2f')

    progress = ProgressMeter(
        len(loader),
        [meters.acc, meters.rob],
        prefix=f"{cfg.DATASET.NAME} ({args.dataset if args.dataset else 'Train Dist'})")

    eps = cfg.AT.EPS
    alpha = float(eps) / 4.0
    steps = 10

    if args.attack == 'aa':
        attack = AutoAttack(model,
                            norm='Linf',
                            eps=eps,
                            version='standard',
                            verbose=False,
                            device=device)
    elif args.attack == 'pgd':
        attack = PGD(model,
                     eps=eps, alpha=alpha, steps=steps, random_start=True)
    elif args.attack == 'tpgd':
        attack = TPGD(model,
                      eps=eps, alpha=alpha, steps=steps)
    elif args.attack == 'cw':
        pass # Use custom pgd with CWLoss later in the loop
    else:
        print(f"Warning: Unknown attack method '{args.attack}'. No attack will be performed.")
        attack = None # Indicate no standard attack object

    print(f"Starting evaluation on {cfg.DATASET.NAME} ({args.dataset if args.dataset else 'Train Dist'})")
    print(f"Attack: {args.attack}, Eps: {eps}, Steps: {steps}")
    print(f"Batch size: {loader.batch_size}, Num workers: {loader.num_workers}")
    print(f"Classifier: {'Linear Probe' if args.linear_probe else 'Zero-Shot BLIP'}")
    if not args.linear_probe:
        print(f"Zero-shot prompt template: '{classify_prompt}'")


    for i, data in enumerate(loader, start=1):
        try:
            imgs, tgts = data['img'], data['label']
        except (TypeError, KeyError):
            try:
                imgs, tgts = data[0], data[1]
            except Exception as e:
                print(f"Error unpacking data batch: {e}. Skipping batch.")
                continue

        imgs, tgts = imgs.to(device), tgts.to(device)
        bs = imgs.size(0)

        with torch.no_grad():
             output = model(imgs)
        acc = accuracy(output, tgts)
        meters.acc.update(acc[0].item(), bs)

        if attack is not None:
            try:
                if args.attack == 'aa':
                     # Handle potential batch size issues with AA's internal eval
                     eval_bs = bs if bs <= 100 else 100
                     adv = attack.run_standard_evaluation(imgs, tgts, bs=eval_bs)
                elif args.attack in ['pgd', 'tpgd']:
                     adv = attack(imgs, tgts)
                # Add other standard attacks here if needed
            except Exception as e:
                print(f"Error during attack {args.attack} execution: {e}. Using clean images for robustness check.")
                adv = imgs.clone().detach() # Fallback to clean images on error
        elif args.attack == 'cw':
             try:
                adv, _ = pgd(imgs, tgts, model, CWLoss, eps, alpha, steps)
             except Exception as e:
                print(f"Error during custom CW-PGD attack: {e}. Using clean images for robustness check.")
                adv = imgs.clone().detach() # Fallback
        else:
             adv = imgs.clone().detach() # No attack specified or unknown attack

        with torch.no_grad():
            output_adv = model(adv)
        rob = accuracy(output_adv, tgts)
        meters.rob.update(rob[0].item(), bs)

        if i == 1 or i % 10 == 0 or i == len(loader):
            progress.display(i)

    print(f"--- Evaluation Summary for {cfg.DATASET.NAME} ({args.dataset if args.dataset else 'Train Dist'}) ---")
    print(f"Clean Accuracy: {meters.acc.avg:.4f}")
    print(f"Robust Accuracy ({args.attack.upper()}): {meters.rob.avg:.4f}")

    current_eval_dataset = args.dataset if args.dataset else train_dataset
    if current_eval_dataset not in result:
        result[current_eval_dataset] = Dict()

    tune_key = 'linear_probe' if args.linear_probe else 'zero_shot'
    if tune_key not in result[current_eval_dataset]:
         result[current_eval_dataset][tune_key] = Dict()

    result[current_eval_dataset][tune_key].clean = meters.acc.avg
    result[current_eval_dataset][tune_key][args.attack] = meters.rob.avg

    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w+') as f:
            yaml.dump(result.to_dict(), f, default_flow_style=False, sort_keys=False)
        print(f'Result saved at: {save_path}')
    except Exception as e:
        print(f"Error saving results to {save_path}: {e}")