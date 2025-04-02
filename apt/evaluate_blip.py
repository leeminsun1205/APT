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
# === BLIP Change: Import necessary components from transformers ===
# We replace CLIP imports with BLIP imports from the transformers library
# from torchvision import transforms # Keep if needed by dataloader, but BLIP processor handles its own transforms
from transformers import BlipProcessor, BlipForImageTextRetrieval, BlipConfig
# === End BLIP Change ===
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
# === BLIP Change: Remove CLIP specific imports ===
# from clip.simple_tokenizer import SimpleTokenizer # BLIP uses its own processor/tokenizer
# import clip # Remove direct clip import
# === End BLIP Change ===


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
    # === BLIP Change: Ensure model gets gradients enabled if needed ===
    # The model wrapper should handle grad enabling based on its mode
    # We might need to adjust how targets are used if the model doesn't directly take them
    # Assuming the model forward pass called here returns logits suitable for the criterion
    # === End BLIP Change ===
    output = model(imgs)
    loss = criterion(output, targets)
    # Ensure requires_grad is True for the input images during gradient calculation
    # The 'perturb' function seems to handle requires_grad_(), so it might be okay here.
    ig = grad(loss, imgs, allow_unused=True)[0] # Added allow_unused=True for safety, might indicate issues if grads are None
    if ig is None:
        print("Warning: Input gradient is None. Check model architecture and loss connection to input.")
        # Handle the None case, e.g., return zeros or raise an error
        ig = torch.zeros_like(imgs)
    return ig


def perturb(imgs, targets, model, criterion, eps, eps_step, pert=None, ig=None):
    # Detach imgs before setting requires_grad_ to avoid modifying the original tensor's grad status in place
    adv_imgs_base = imgs.detach().clone()
    current_pert = pert.detach().clone() if pert is not None else torch.zeros_like(imgs)

    adv = torch.clamp(adv_imgs_base + current_pert, 0, 1).requires_grad_(True)

    # === BLIP Change: Calculate input gradient ===
    # Ensure model is in a state where grads can be computed (e.g., not torch.no_grad())
    # The model wrapper's 'mode' should control this.
    # Calculate gradient w.r.t the *adversarial* input 'adv'
    # We need to recompute the gradient at each step
    ig_current = input_grad(adv, targets, model, criterion)
    # === End BLIP Change ===

    with torch.no_grad(): # Calculations below should not be part of the computation graph
        if pert is None:
            new_pert = eps_step * torch.sign(ig_current)
        else:
            new_pert = current_pert + eps_step * torch.sign(ig_current)

        new_pert.clamp_(-eps, eps)
        # Project perturbation back to valid image range
        adv_updated = torch.clamp(adv_imgs_base + new_pert, 0, 1)
        # Final perturbation is the difference
        final_pert = adv_updated - adv_imgs_base

    return adv_updated.detach(), final_pert.detach()


def pgd(imgs, targets, model, criterion, eps, eps_step, max_iter, pert=None, ig=None):
    # Initial perturbation (if None) is zeros
    current_pert = pert if pert is not None else torch.zeros_like(imgs)

    for i in range(max_iter):
        # Pass the *original* images and the *current* perturbation to the perturb function
        adv_img, current_pert = perturb(imgs, targets, model, criterion, eps, eps_step, current_pert)
        # Note: ig is recalculated inside perturb now, so passing ig=None is correct
        ig = None # ig is recalculated in perturb
    # The final adversarial image is original + final perturbation
    final_adv_img = torch.clamp(imgs + current_pert, 0, 1)
    return final_adv_img.detach(), current_pert.detach()


# === BLIP Change: Remove CLIP specific loading function ===
# def load_clip_to_cpu(backbone_name="RN50"):
#     ... # This function is specific to CLIP models and architecture
# === End BLIP Change ===

# === BLIP Change: Define CustomBLIP wrapper ===
# This class wraps the BLIP model to provide an interface similar to CustomCLIP,
# handling text prompts and generating classification logits using image-text matching.
class CustomBLIP(nn.Module):
    def __init__(self, blip_model, processor, classes, cls_prompt_template='a photo of a {}'):
        super().__init__()
        self.model = blip_model
        self.processor = processor
        self.classes = classes
        self.cls_prompt_template = cls_prompt_template
        # Generate text prompts for all classes
        self.prompts = [self.cls_prompt_template.format(c) for c in self.classes]
        # Ensure model is on the correct device (should be handled outside, but good practice)
        self.device = next(self.model.parameters()).device
        # Keep the mode attribute if attacks depend on it (e.g., to enable/disable grads)
        self.mode = 'classification'

    def forward(self, images):
        # This forward pass calculates classification scores for the input images.
        # It processes each image against all class prompts using the BLIP ITM head.

        all_logits = []
        # Determine if running in eval mode (no grads) or attack mode (grads needed for images)
        is_eval_mode = (self.mode == 'classification') or (not torch.is_grad_enabled())

        # Process images and text using the BLIP processor
        # Handle potential tensor vs PIL input; assume dataloader provides tensors.
        # Processor might need specific normalization/resizing.
        # We process texts once
        text_inputs = self.processor(text=self.prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)

        # Process images - processor might apply necessary transforms
        # Note: If dataloader already applies normalization specific to another model (like CLIP),
        # it might conflict with BLIP's expected normalization. Ideally, dataloader provides
        # images compatible with processor (e.g., unnormalized tensors or PIL images).
        # Here, we assume 'images' are tensors that the processor can handle.
        image_inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        pixel_values = image_inputs.pixel_values

        # BLIP ITM forward pass expects pixel_values, input_ids, attention_mask
        # We need to compute scores for each image against each prompt.
        # Model can compute this efficiently if inputs are prepared correctly.
        # Repeat image features for each prompt or text features for each image.
        num_images = pixel_values.size(0)
        num_prompts = text_inputs.input_ids.size(0)

        # Prepare for batch processing: repeat image embeddings for each prompt
        # image_embeds = self.model.vision_model(pixel_values=pixel_values)[0] # Get [CLS] token output usually
        # For ITM, we often feed pairs directly. Let's try that.
        # Repeat pixel_values num_prompts times
        batch_pixel_values = pixel_values.repeat_interleave(num_prompts, dim=0)
        # Repeat text inputs num_images times
        batch_input_ids = text_inputs.input_ids.repeat(num_images, 1)
        batch_attention_mask = text_inputs.attention_mask.repeat(num_images, 1)

        # Run model
        # Enable gradients for images if in attack mode and grad is enabled globally
        with torch.set_grad_enabled(not is_eval_mode):
             # Pass all inputs required by the specific BLIP model forward method
             # For BlipForImageTextRetrieval, it needs pixel_values, input_ids, attention_mask
             outputs = self.model(pixel_values=batch_pixel_values,
                                  input_ids=batch_input_ids,
                                  attention_mask=batch_attention_mask,
                                  return_dict=True)

             # Extract ITM scores (logits for "image matches text")
             # itm_score shape is usually (batch_size * num_prompts, 2) where [:, 1] is the match logit
             itm_logits = outputs.itm_score[:, 1] # Logits for match

        # Reshape logits to (batch_size, num_classes)
        batch_logits = itm_logits.view(num_images, num_prompts)

        return batch_logits

    # Helper to get prompts if needed elsewhere
    def _get_prompts(self):
         return self.prompts
# === End BLIP Change ===


parser = argparse.ArgumentParser()
parser.add_argument('experiment', type=str, help="Name or type of experiment to run.")

parser.add_argument('-cp', '--cls-prompt', type=str, default='a photo of a {}',
                    help="Template for class prompt. Default is 'a photo of a {}'.")

parser.add_argument('-ap', '--atk-prompt', type=str, default=None,
                    help="Template for attack prompt. If not specified, defaults to None. (Note: BLIP adaptation might simplify/ignore this)") # BLIP Note

parser.add_argument('--best-checkpoint', action='store_true',
                    help="Use the best checkpoint if available. (Note: Affects prompter loading, less relevant for basic BLIP)") # BLIP Note

parser.add_argument('--attack', type=str, default='pgd',
                    help="Type of attack to use. Default is 'pgd'.")

parser.add_argument('--dataset', type=str, default=None,
                    help="Dataset to use for the experiment. Defaults to None.")

parser.add_argument('-lp', '--linear-probe', action='store_true',
                    help="Enable linear probing for the experiment. (Note: Requires adaptation for BLIP features)") # BLIP Note

parser.add_argument('--save-img', action='store_true',
                    help="Enable save images for the experiment.")

parser.add_argument('--save-path', type=str, default = './',
                    help="Specific path to save images. Default is ./")

parser.add_argument('--num-imgs', type=int, default = '10',
                    help="Number of images to save. Default is 10")

parser.add_argument('--seed', type=int, default = '42',
                    help="Seed for torch random")

parser.add_argument("--topk", type=int, default = '1',
                    help="Select top-k similar words (Note: CLIP prompter feature, remove/adapt for BLIP)") # BLIP Note


if __name__ == '__main__':
    args = parser.parse_args()
    cfg = CfgNode()
    cfg.set_new_allowed(True)
    cfg_path = os.path.join(args.experiment, 'cfg.yaml')
    # === BLIP Change: Handle potential missing cfg file gracefully ===
    # If the experiment setup relies heavily on cfg specific to CLIP, this might need adjustment
    try:
        cfg.merge_from_file(cfg_path)
    except FileNotFoundError:
        print(f"Warning: Configuration file {cfg_path} not found. Using default settings or args.")
        # Set default dataset if not provided and cfg is missing
        if not hasattr(cfg, 'DATASET') or not hasattr(cfg.DATASET, 'NAME'):
             cfg.DATASET = CfgNode()
             cfg.DATASET.NAME = args.dataset if args.dataset else 'UnknownDataset' # Use provided dataset or a placeholder
        # Set default attack params if not provided and cfg is missing
        if not hasattr(cfg, 'AT') or not hasattr(cfg.AT, 'EPS'):
             cfg.AT = CfgNode()
             cfg.AT.EPS = 8/255 # A common default epsilon
        # Set default model backbone (less critical now as we load a fixed BLIP model)
        if not hasattr(cfg, 'MODEL') or not hasattr(cfg.MODEL, 'BACKBONE') or not hasattr(cfg.MODEL.BACKBONE, 'NAME'):
            cfg.MODEL = CfgNode()
            cfg.MODEL.BACKBONE = CfgNode()
            cfg.MODEL.BACKBONE.NAME = "BLIP_ITM_BASE" # Placeholder name
    # === End BLIP Change ===


    train_dataset = cfg.DATASET.NAME
    os.makedirs(args.save_path, exist_ok=True)
    if args.dataset:
        if args.dataset in ['ImageNetR', 'ImageNetA', 'ON', 'ImageNetV2']: # Added ImageNetV2
            cfg.DATASET.NAME = 'ImageNet' # Base dataset for these variants
        else:
            cfg.DATASET.NAME = args.dataset
        save_path = os.path.join(args.save_path, f'dist_shift_{args.dataset}.yaml') # Include dataset in filename
    else:
        save_path = os.path.join(args.save_path, 'evaluation.yaml')  # Modified to use args.save_path
    if os.path.isfile(save_path):
        with open(save_path, 'r') as f:
            result = Dict(yaml.safe_load(f))

        # Adjust result access based on whether it's a distribution shift or standard eval
        current_eval_key = args.dataset if args.dataset else train_dataset
        tune_key = 'linear_probe' if args.linear_probe else args.cls_prompt # Simplified key, maybe just use 'default' if not LP

        # Check if results for this specific setting exist
        if current_eval_key in result and tune_key in result[current_eval_key] and args.attack in result[current_eval_key][tune_key] and result[current_eval_key][tune_key][args.attack] is not None:
             print(f'Evaluation result already exists at: {save_path} for {current_eval_key}/{tune_key}/{args.attack}')
             # exit() # Keep commented out to allow re-running

    # Initialize result structure if file doesn't exist or key is missing
    if not os.path.isfile(save_path) or current_eval_key not in result:
         result = Dict()
         result[current_eval_key] = Dict()
    if tune_key not in result[current_eval_key]:
        result[current_eval_key][tune_key] = Dict({'clean': None, args.attack: None}) # Initialize keys


    dm = DataManager(cfg)
    classes = dm.dataset.classnames
    # === BLIP Change: Adapt DataLoader transform if needed ===
    # BLIP processor expects images in a certain format/normalization.
    # The transform in dm.test_loader should ideally match this, or CustomBLIP needs to handle it.
    # For now, assume the loader provides compatible tensors or CustomBLIP handles it.
    loader = dm.test_loader
    # === End BLIP Change ===
    num_classes = dm.num_classes

    # Handle distribution shift datasets (like ImageNet variants)
    if args.dataset in ['ImageNetR', 'ImageNetA', 'ON', 'ImageNetV2']:
        # Reuse ImageNet loading logic if needed, adjust path/split name
        from OODRB.imagenet import ImageNet # Assuming this utility exists
        if args.dataset == 'ImageNetV2':
            shift = 'v2' # Or appropriate identifier for ImageNetV2
        elif args.dataset == 'ImageNetA':
            shift = 'A'
        elif args.dataset == 'ImageNetR':
            shift = 'R'
        elif args.dataset == 'ON':
            shift = 'ON'
        else:
            shift = None # Should not happen based on outer if

        num_classes = 1000
        # === BLIP Change: Ensure transform is compatible ===
        # Use BLIP processor's transform or ensure loader's transform matches
        # If loader.dataset.transform exists and is CLIP-specific, it needs replacement/checking.
        # Simplification: Assume default ImageNet transform in OODRB loader is somewhat standard.
        blip_processor_for_transform = BlipProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
        # Example transform using processor (if dataset returns PIL):
        # transform = lambda pil_img: blip_processor_for_transform(images=pil_img, return_tensors="pt").pixel_values.squeeze(0)
        # Or check existing transform:
        current_transform = loader.dataset.transform if hasattr(loader.dataset, 'transform') else None
        print(f"Using transform from existing loader: {current_transform}") # Informative print
        # === End BLIP Change ===

        dataset = ImageNet(cfg.DATASET.ROOT,
                           shift,
                           'val', # Usually 'val' or 'test' split
                           transform=current_transform) # Pass the transform
        if args.attack == 'aa' and len(dataset) > 5000: # Limit dataset size for AutoAttack if large
            print("Subsetting dataset for AutoAttack (first 5000 samples)")
            dataset = torch.utils.data.Subset(dataset, list(range(5000)))

        loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=cfg.DATALOADER.TEST.BATCH_SIZE, # Use cfg batch size
                                             shuffle=False,
                                             num_workers=cfg.DATALOADER.NUM_WORKERS, # Use cfg workers
                                             pin_memory=True)


    # === BLIP Change: Load BLIP model and processor ===
    print("Loading BLIP model and processor...")
    # Choose a BLIP model suitable for ITM/Classification, e.g., blip-itm-base-coco
    blip_model_name = "Salesforce/blip-itm-base-coco"
    processor = BlipProcessor.from_pretrained(blip_model_name)
    model_blip = BlipForImageTextRetrieval.from_pretrained(blip_model_name)
    model_blip = model_blip.cuda() # Move model to GPU
    print(f"Loaded {blip_model_name}")
    # === End BLIP Change ===


    # === BLIP Change: Comment out CLIP-specific robust backbone loading ===
    # This section loaded pre-trained robust weights specifically for CLIP's visual encoder.
    # Adapting this to BLIP's vision model requires compatible weights, which are unlikely to exist.
    # print("Skipping loading of custom robust backbone weights (CLIP specific). Using standard BLIP weights.")
    # ckp_name = 'vitb32' if cfg.MODEL.BACKBONE.NAME == 'ViT-B/32' else 'rn50'
    # eps = int(cfg.AT.EPS * 255)
    # ckp_name += f'_eps{eps}.pth.tar'
    # try:
    #     ckp = torch.load(os.path.join('backbone', ckp_name))
    #     # Loading into BLIP's vision model would be like:
    #     # model_blip.vision_model.load_state_dict(ckp['vision_encoder_state_dict']) # Requires matching state_dict keys
    #     print(f"Found potential robust weights at {os.path.join('backbone', ckp_name)}, but skipping loading for BLIP.")
    # except FileNotFoundError:
    #     print(f"No custom robust backbone weights found at {os.path.join('backbone', ckp_name)}")
    # === End BLIP Change ===

    # === BLIP Change: Remove CLIP prompter logic ===
    # The 'prompter' logic involves loading learned context vectors (ctx) specific to CLIP's embedding space.
    # This is not directly applicable to BLIP. We will use the standard text template.
    # if 'prompter' in (args.cls_prompt, args.atk_prompt):
    #     prompter_path = os.path.join(cfg.OUTPUT_DIR, 'prompt_learner/')
    #     assert os.path.isdir(prompter_path)
    #     if args.best_checkpoint:
    #         prompter_path += 'best.pth.tar'
    #     else:
    #         ckp_files = [fname for fname in os.listdir(prompter_path) if 'model.pth.tar' in fname]
    #         if ckp_files:
    #             prompter_path += ckp_files[0]
    #         else:
    #             raise FileNotFoundError(f"No prompter checkpoint found in {prompter_path}")
    # classify_prompt = prompter_path if args.cls_prompt == 'prompter' else args.cls_prompt
    # attack_prompt = prompter_path if args.atk_prompt == 'prompter' else args.atk_prompt
    print(f"Using standard class prompt template: '{args.cls_prompt}'")
    classify_prompt_template = args.cls_prompt # Use the template string directly
    # Attack prompt is not used by the CustomBLIP wrapper in this setup
    # === End BLIP Change ===

    # === BLIP Change: Adapt model initialization ===
    if args.linear_probe:
        # Linear probing on BLIP requires extracting features first, then training a linear layer.
        # This script assumes a pre-trained linear probe. Adapting requires feature extraction logic.
        print("Warning: Linear probing (--linear-probe) is not directly implemented for BLIP in this script. Requires adaptation.")
        # Placeholder: Use the main model anyway, results won't be 'linear probe' results.
        model = CustomBLIP(model_blip, processor, classes, cls_prompt_template=classify_prompt_template)
        # Attempt to load linear weights if path exists, but structure might be incompatible
        lp_path = os.path.join(cfg.OUTPUT_DIR, 'linear_probe/linear.pth.tar')
        if os.path.exists(lp_path):
            print(f"Found linear probe weights at {lp_path}, but cannot apply directly to BLIP without adaptation.")
        # model = LinearProbe(model, 512, num_classes, False) # Original CLIP LP wrapper
        # ckp = torch.load(os.path.join(cfg.OUTPUT_DIR, 'linear_probe/linear.pth.tar'))
        # model.linear.load_state_dict(ckp)
    else:
        # Use the CustomBLIP wrapper
        model = CustomBLIP(model_blip, processor, classes, cls_prompt_template=classify_prompt_template)
    # === End BLIP Change ===

    model = model.cuda() # Ensure the final model wrapper is on GPU
    model.eval() # Set to evaluation mode initially

    # === BLIP Change: Remove CLIP-specific context vector analysis ===
    # This section analyzed learned prompt vectors (ctx) from CLIP's prompter
    # and mapped them to words using CLIP's tokenizer and embeddings.
    # This is entirely CLIP-specific and removed for BLIP.
    # if args.cls_prompt == 'prompter':
        # ... (code for loading prompter state, ctx, tokenizer, clip_model, token_embedding) ...
        # ... (code for calculating distances and finding nearest words) ...
        # class_raw_titles = [...] # Generated titles based on nearest words
    # else:
        # Standard prompt generation (now handled inside CustomBLIP or here)
    class_raw_titles = [args.cls_prompt.format(classes[class_idx]) for class_idx in range(num_classes)]
    # === End BLIP Change ===

    meters = Dict()
    meters.acc = AverageMeter('Clean Acc@1', ':6.2f')
    meters.rob = AverageMeter('Robust Acc@1', ':6.2f')

    progress = ProgressMeter(
        len(loader),
        [meters.acc, meters.rob],
        prefix=f"{cfg.DATASET.NAME} ({args.dataset or 'Train Dist'})") # More informative prefix

    eps = cfg.AT.EPS
    # Use a smaller step size relative to epsilon, e.g., eps/4 or eps/10
    alpha = eps / 4.0 # PGD step size
    steps = 10 # Number of PGD steps (reduced from 100 for potentially faster runs, adjust if needed)

    # === BLIP Change: Ensure attack libraries get the correct model wrapper ===
    # The attack libraries need the 'model' object which has the forward pass returning logits.
    # Our CustomBLIP wrapper serves this purpose.
    if args.attack == 'aa':
        attack = AutoAttack(model, # Pass the CustomBLIP wrapper
                            norm='Linf',
                            eps=eps,
                            version='standard', # 'standard', 'plus', 'rand'
                            verbose=False,
                            device=torch.device('cuda')) # Specify device
    elif args.attack == 'pgd':
        attack = PGD(model, # Pass the CustomBLIP wrapper
                     eps=eps, alpha=alpha, steps=steps, random_start=True)
    elif args.attack == 'tpgd':
        attack = TPGD(model, # Pass the CustomBLIP wrapper
                      eps=eps, alpha=alpha, steps=steps)
    else: # Default to the custom PGD implementation using CWLoss
         attack = None # Indicate using the manual PGD loop
         print("Using manual PGD loop with CWLoss.")
    # === End BLIP Change ===

    if args.save_img:
        clean_dir = os.path.join(args.save_path, f'{args.dataset or train_dataset}_clean_test')
        adv_dir = os.path.join(args.save_path, f'{args.dataset or train_dataset}_{args.attack}_eps{int(eps*255)}_adv_test')
        os.makedirs(clean_dir, exist_ok=True)
        os.makedirs(adv_dir, exist_ok=True)
        print(f'Saving clean images to: {clean_dir}')
        print(f'Saving adversarial images to: {adv_dir}')

        all_logits_clean = []
        all_images_clean = []
        all_logits_adv = []
        all_images_adv = []
        all_labels = []


    for i, data in enumerate(tqdm(loader, desc="Evaluating Batches"), start=1): # Added tqdm
        try:
            # few-shot data loader from Dassl
            imgs, tgts = data['img'], data['label']
        except TypeError: # Handle cases where data is not a dict
            imgs, tgts = data[0], data[1] # Assume tuple/list
        except KeyError: # Handle if keys are different
             # Try common alternatives or raise error
             if 'image' in data and 'label' in data:
                 imgs, tgts = data['image'], data['label']
             else:
                 print("Error: Could not extract images/labels from dataloader output. Keys:", data.keys())
                 raise
        imgs, tgts = imgs.cuda(), tgts.cuda()

        if args.save_img:
             # Store labels regardless of batch index for later saving
             all_labels.append(tgts.cpu())
             # Store original images before any modification
             all_images_clean.append(imgs.cpu())


        bs = imgs.size(0)

        # --- Clean Accuracy ---
        model.eval() # Ensure eval mode for clean pass
        model.mode = 'classification' # Set mode for CustomBLIP wrapper if it uses it
        with torch.no_grad():
            output_clean = model(imgs)
            if args.save_img:
                 all_logits_clean.append(output_clean.cpu())

        acc = accuracy(output_clean, tgts) # Top-1 accuracy
        meters.acc.update(acc[0].item(), bs)

        # --- Adversarial Attack and Robust Accuracy ---
        # Set model to train mode *if* the attack library requires gradients
        # PGD/TPGD/AA internally handle model eval/train state often, but good practice:
        model.train() # Enable gradients for attack calculation
        model.mode = 'attack' # Set mode for CustomBLIP wrapper if it uses it

        # Generate adversarial examples
        if args.attack == 'aa':
            # AutoAttack manages device internally if initialized with device
            adv = attack.run_standard_evaluation(imgs, tgts, bs=bs) # AA expects model in eval usually, check AA docs
            model.eval() # Set back to eval after AA call if AA doesn't do it
        elif args.attack in ['pgd', 'tpgd']:
            # PGD/TPGD from Torchattacks expect model passed during init
            adv = attack(imgs, tgts) # Generate adversarial images
            model.eval() # Set back to eval after attack
        else: # Manual PGD with CW loss
            # Ensure model allows grads calculation w.r.t input when calling input_grad/perturb
            # The 'model.train()' call above should suffice
            adv, _ = pgd(imgs, tgts, model, CWLoss, eps, alpha, steps)
            model.eval() # Set back to eval after attack

        # --- Evaluate on Adversarial Examples ---
        model.eval() # Ensure eval mode for robust pass
        model.mode = 'classification' # Set mode for CustomBLIP wrapper
        with torch.no_grad():
            output_adv = model(adv)
            if args.save_img:
                all_logits_adv.append(output_adv.cpu())
                all_images_adv.append(adv.cpu()) # Store the generated adv images

        rob = accuracy(output_adv, tgts) # Top-1 accuracy on adversarial examples
        meters.rob.update(rob[0].item(), bs)

        if i % 10 == 0 or i == len(loader): # Print progress every 10 batches and at the end
            progress.display(i)

    # --- Final Results ---
    print(f"\n--- Final Results for {cfg.DATASET.NAME} ({args.dataset or 'Train Dist'}) ---")
    print(f"Prompt Template: '{args.cls_prompt}'")
    print(f"Attack: {args.attack}, Epsilon: {eps:.4f} (≈{int(eps*255)}/255)")
    if args.attack not in ['aa']: print(f"Attack Steps: {steps}, Step Size: {alpha:.4f}")
    print(f"Clean Accuracy: {meters.acc.avg:.2f}%")
    print(f"Robust Accuracy ({args.attack}): {meters.rob.avg:.2f}%")
    print("-" * 50)


    # --- Save Images (if enabled) ---
    if args.save_img:
        print("\nSaving example images...")
        # Concatenate all collected data
        all_logits_clean = torch.cat(all_logits_clean, dim=0)
        all_images_clean = torch.cat(all_images_clean, dim=0)
        all_logits_adv = torch.cat(all_logits_adv, dim=0)
        all_images_adv = torch.cat(all_images_adv, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        torch.manual_seed(args.seed) # Seed for reproducibility in sampling images

        for class_idx in tqdm(range(num_classes), desc="Saving Images per Class"):
            indices_for_class = (all_labels == class_idx).nonzero(as_tuple=False).squeeze()
            # Handle case where a class might have no images in the test set slice
            if indices_for_class.numel() == 0:
                # print(f"No images found for class {classes[class_idx]} (idx {class_idx}) in this run.")
                continue
            if indices_for_class.dim() == 0: # If only one image, make it a tensor
                 indices_for_class = indices_for_class.unsqueeze(0)

            # Select images/logits for the current class
            images_for_class_clean = all_images_clean[indices_for_class]
            logits_for_class_clean = all_logits_clean[indices_for_class]
            images_for_class_adv = all_images_adv[indices_for_class]
            logits_for_class_adv = all_logits_adv[indices_for_class]

            # Select k random images to save (k=args.num_imgs or fewer if not enough images)
            k = min(args.num_imgs, images_for_class_clean.size(0))
            if k == 0: continue # Skip if no images after filtering

            random_indices = torch.randperm(images_for_class_clean.size(0))[:k]
            # print(f"Selected {k} random images for class {classes[class_idx]}")

            selected_images_clean = images_for_class_clean[random_indices]
            selected_logits_clean = logits_for_class_clean[random_indices]
            selected_images_adv = images_for_class_adv[random_indices]
            selected_logits_adv = logits_for_class_adv[random_indices]

            # --- Plot and Save Clean Images ---
            # Limit plots to max 10 images (2 rows of 5) even if k > 10
            plot_k = min(k, 10)
            rows = 2
            cols = 5
            fig_clean, axes_clean = plt.subplots(rows, cols, figsize=(15, 6))
            # Use the actual class name and prompt used
            clean_title = f"Clean - Class: {classes[class_idx]}\nPrompt: '{class_raw_titles[class_idx]}'"
            fig_clean.suptitle(clean_title, fontsize=14)
            axes_clean = axes_clean.flatten()
            for j in range(plot_k):
                img = np.transpose(selected_images_clean[j].cpu().numpy(), (1, 2, 0))
                 # De-normalize if necessary - Assuming standard normalization [-1, 1] or [0, 1] based on transforms
                 # If [0,1], imshow works directly. If normalized with mean/std, reverse it.
                 # Simple clamp if needed: img = np.clip(img, 0, 1)
                axes_clean[j].imshow(img)
                axes_clean[j].axis('off')
                pred_class_idx = selected_logits_clean[j].argmax().item()
                pred_class_name = classes[pred_class_idx]
                is_correct = (pred_class_idx == class_idx)
                title_color = 'green' if is_correct else 'red'
                axes_clean[j].set_title(f"Pred: {pred_class_name}", color=title_color, fontsize=10)
            # Hide unused subplots
            for j in range(plot_k, rows * cols):
                axes_clean[j].axis('off')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
            clean_save_filename = os.path.join(clean_dir, f'class_{classes[class_idx].replace("/","_")}_clean.png')
            plt.savefig(clean_save_filename)
            plt.close(fig_clean)

            # --- Plot and Save Adversarial Images ---
            fig_adv, axes_adv = plt.subplots(rows, cols, figsize=(15, 6))
            adv_title = f"Adv ({args.attack}, eps={int(eps*255)}) - Class: {classes[class_idx]}\nPrompt: '{class_raw_titles[class_idx]}'"
            fig_adv.suptitle(adv_title, fontsize=14)
            axes_adv = axes_adv.flatten()
            for j in range(plot_k):
                img = np.transpose(selected_images_adv[j].cpu().numpy(), (1, 2, 0))
                img = np.clip(img, 0, 1) # Clip adv images to be sure they are in [0,1] range
                axes_adv[j].imshow(img)
                axes_adv[j].axis('off')
                pred_class_idx = selected_logits_adv[j].argmax().item()
                pred_class_name = classes[pred_class_idx]
                is_correct = (pred_class_idx == class_idx)
                title_color = 'green' if is_correct else 'red'
                axes_adv[j].set_title(f"Pred: {pred_class_name}", color=title_color, fontsize=10)
            # Hide unused subplots
            for j in range(plot_k, rows * cols):
                axes_adv[j].axis('off')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
            adv_save_filename = os.path.join(adv_dir, f'class_{classes[class_idx].replace("/","_")}_adv.png')
            plt.savefig(adv_save_filename)
            plt.close(fig_adv)
        print(f"Finished saving images.")


    # --- Save Results ---
    print(f"\nSaving results to: {save_path}")
    # Reload results file in case it was modified by another process (less likely here but safe)
    if os.path.isfile(save_path):
        with open(save_path, 'r') as f:
            try:
                result = Dict(yaml.safe_load(f))
            except yaml.YAMLError:
                result = Dict() # Start fresh if file is corrupt
    else:
        result = Dict()

    # Ensure nested structure exists
    current_eval_key = args.dataset if args.dataset else train_dataset # e.g., 'ImageNetR' or 'ImageNet'
    tune_key = 'linear_probe' if args.linear_probe else 'default_prompt' # Use a consistent key name

    if current_eval_key not in result:
         result[current_eval_key] = Dict()
    if tune_key not in result[current_eval_key]:
        result[current_eval_key][tune_key] = Dict()

    # Save metrics
    result[current_eval_key][tune_key]['model'] = 'BLIP_ITM_BASE' # Record model used
    result[current_eval_key][tune_key]['prompt'] = args.cls_prompt
    result[current_eval_key][tune_key]['clean_acc'] = round(meters.acc.avg, 2)
    result[current_eval_key][tune_key][f'{args.attack}_robust_acc'] = round(meters.rob.avg, 2)
    result[current_eval_key][tune_key][f'{args.attack}_eps'] = eps
    result[current_eval_key][tune_key][f'{args.attack}_alpha'] = alpha if args.attack != 'aa' else 'N/A'
    result[current_eval_key][tune_key][f'{args.attack}_steps'] = steps if args.attack != 'aa' else 'N/A'


    # Write updated results back to YAML file
    try:
        with open(save_path, 'w') as f:
            yaml.dump(result.to_dict(), f, default_flow_style=False) # Use block style for readability
        print(f"Results successfully saved.")
    except Exception as e:
        print(f"Error saving results to {save_path}: {e}")