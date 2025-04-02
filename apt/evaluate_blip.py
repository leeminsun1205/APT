import os
import torch
from yacs.config import CfgNode
import yaml
import argparse

from torchvision.datasets import *
# --- MODIFICATION START: Import necessary Hugging Face classes ---
from transformers import BlipProcessor, BlipModel
from PIL import Image # Thường cần thiết khi làm việc với processor ảnh
# --- MODIFICATION END ---

from torch.autograd import grad, Variable

from addict import Dict

from dassl.data import DataManager
# from blip.blip import blip_feature_extractor # Removed: Using HF implementation
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

# --- MODIFICATION START: Define a wrapper for Hugging Face BLIP Zero-Shot ---
class CustomBLIP(torch.nn.Module):
    def __init__(self, processor, model, classnames, cls_prompt_template="a photo of a {}"):
        super().__init__()
        self.processor = processor
        self.model = model
        self.classnames = classnames
        self.cls_prompt_template = cls_prompt_template
        # Get device from model
        self.device = next(model.parameters()).device

        # Pre-compute text features for zero-shot classification
        self.text_prompts = [self.cls_prompt_template.format(c) for c in self.classnames]
        # Process text prompts
        text_inputs = self.processor(text=self.text_prompts, return_tensors="pt", padding=True)
        # Move text inputs to the correct device
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}

        # Get text features
        with torch.no_grad():
            self.text_features = self.model.get_text_features(**text_inputs)
            # Optional: Normalize features (common practice in contrastive models like CLIP/BLIP)
            # self.text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)

    def forward(self, image):
        # Assume 'image' is a batch of image tensors (pixel_values) from the dataloader
        # Ensure image is on the correct device
        pixel_values = image.to(self.device)

        # Get image features
        # No gradient needed for standard inference, but attacks might require it later
        # Let the attack library handle requires_grad_() on the input image
        image_features = self.model.get_image_features(pixel_values=pixel_values)
        # Optional: Normalize image features
        # image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Compute zero-shot logits (similarity scores)
        # Ensure text_features are accessible and on the correct device (handled in __init__)
        logits_per_image = image_features @ self.text_features.t() # Use transpose .t()

        # Optional: Apply learned temperature scaling if the model has it (like CLIP)
        # Usually accessed via model.logit_scale.exp() - BLIP might not expose it this way directly
        # Check BLIP model documentation if scaling is needed/available
        # logits_per_image = logits_per_image * self.model.logit_scale.exp()

        return logits_per_image # Return logits for compatibility with accuracy function and attacks

    # Removed mode switching ('attack'/'classification') as the forward pass is now consistent for ZS
    # The attack libraries will modify the input image and call this forward method
# --- MODIFICATION END ---

# Input grad and perturb functions remain largely the same,
# but their effectiveness might depend on how well BLIP's ZS logits work with gradient-based attacks
def input_grad(imgs, targets, model, criterion):
    # Ensure model is in eval mode for consistency, but allow gradients for attack
    # model.eval() # Might interfere with gradient calculation needed by attack
    imgs.requires_grad_(True)
    output = model(imgs)
    loss = criterion(output, targets)
    # Zero gradients before calculating new ones
    model.zero_grad()
    # Calculate gradients
    ig = grad(loss, imgs, retain_graph=False)[0] # retain_graph=False usually safe here
    imgs.requires_grad_(False) # Detach after use
    return ig

def perturb(imgs, targets, model, criterion, eps, eps_step, pert=None, ig=None):
    # Use a clone of imgs to avoid modifying the original tensor directly during grad calculation
    imgs_clone = imgs.clone().detach()
    adv = imgs_clone if pert is None else torch.clamp(imgs_clone+pert, 0, 1)
    adv.requires_grad_(True)

    # Get input gradient
    ig_calc = input_grad(adv, targets, model, criterion) if ig is None else ig

    # Detach adv after gradient calculation if it's not needed anymore
    adv.requires_grad_(False)

    # Calculate perturbation step
    if pert is None:
        pert = eps_step * torch.sign(ig_calc)
    else:
        pert += eps_step * torch.sign(ig_calc)

    # Clamp perturbation and create adversarial example
    pert = torch.clamp(pert, -eps, eps)
    adv_final = torch.clamp(imgs_clone + pert, 0, 1) # Apply perturbation to the original clone
    # Ensure the final adversarial example retains requires_grad state if needed later,
    # but usually detached for evaluation. Calculate the final perturbation relative to original imgs.
    final_pert = adv_final.detach() - imgs.detach()

    return adv_final.detach(), final_pert.detach()


def pgd(imgs, targets, model, criterion, eps, eps_step, max_iter, pert=None, ig=None):
    # PGD requires iterative gradient updates
    imgs_perturbed = imgs.clone().detach() # Start with clean images
    current_pert = pert.clone().detach() if pert is not None else torch.zeros_like(imgs).detach()

    for i in range(max_iter):
        # Calculate gradient on the *current* adversarial example
        adv_for_grad = imgs_perturbed.requires_grad_(True)
        output = model(adv_for_grad)
        loss = criterion(output, targets)
        model.zero_grad() # Zero gradients for the model
        # Calculate gradient w.r.t the input
        ig_step = grad(loss, adv_for_grad, retain_graph=False)[0]
        adv_for_grad.requires_grad_(False) # Detach

        # Update perturbation
        current_pert += eps_step * torch.sign(ig_step.detach())
        current_pert = torch.clamp(current_pert, -eps, eps) # Clamp perturbation budget

        # Apply perturbation to the original clean image and clamp pixel values
        imgs_perturbed = torch.clamp(imgs + current_pert, 0, 1).detach()

    final_pert = imgs_perturbed - imgs
    return imgs_perturbed, final_pert



parser = argparse.ArgumentParser()
parser.add_argument('experiment')
# --- MODIFICATION START: Adjusted default prompt for better ZS ---
parser.add_argument('-cp','--cls-prompt', default='a photo of a {}') # Standard ZS prompt
# --- MODIFICATION END ---
parser.add_argument('-ap','--atk-prompt', default=None) # Less relevant for standard image attacks in ZS
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
            # --- MODIFICATION START: Ensure base dataset name is correct for DataManager ---
            cfg.DATASET.NAME = 'ImageNet'
            # --- MODIFICATION END ---
        else:
            cfg.DATASET.NAME = args.dataset
        save_path = os.path.join(cfg.OUTPUT_DIR, f'dist_shift_{args.dataset}.yaml') # Make path dataset-specific
    else:
        save_path = os.path.join(cfg.OUTPUT_DIR, 'evaluation.yaml')

    # --- MODIFICATION START: Adjusted result structure key ---
    if os.path.isfile(save_path):
        with open(save_path, 'r') as f:
            result = Dict(yaml.safe_load(f))

        # Handle nested structure for distribution shift datasets
        current_eval_dataset = args.dataset if args.dataset else train_dataset
        if current_eval_dataset not in result:
             result[current_eval_dataset] = Dict()

        tune_key = 'linear_probe' if args.linear_probe else 'zero_shot' # Use 'zero_shot' key
        if tune_key not in result[current_eval_dataset]:
            result[current_eval_dataset][tune_key] = Dict()

        # Check if specific attack result exists
        attack_key = args.attack if args.attack else 'clean' # Assume 'clean' if no attack specified? Or handle separately.
        if attack_key in result[current_eval_dataset][tune_key] and result[current_eval_dataset][tune_key][attack_key] is not None:
            print(f'Eval result for {current_eval_dataset}/{tune_key}/{attack_key} already exists at: {save_path}')
            # exit() # Keep exit commented for testing, uncomment for production
    else:
        result = Dict()
    # --- MODIFICATION END ---

    dm = DataManager(cfg)
    classes = dm.dataset.classnames
    loader = dm.test_loader
    num_classes = dm.num_classes

    if args.dataset in ['ImageNetR', 'ImageNetA', 'ON', 'ImageNetV2'] or (train_dataset == 'ImageNet' and args.dataset is None and args.attack == 'aa'):
        from OODRB.imagenet import ImageNet # Assuming this import works
        if args.dataset == 'ImageNetV2':
            shift = 'v2'
        elif args.dataset == 'ImageNetA':
            shift = 'A'
        elif args.dataset == 'ImageNetR':
            shift = 'R'
        elif args.dataset == 'ON':
            shift = 'ON'
        else: # Default to val if dataset is None but train_dataset was ImageNet
             shift = 'val' # Or None depending on ImageNet class implementation
        num_classes = 1000
        # --- MODIFICATION START: Use the processor's image transform ---
        # Need to load the HF processor first to get the correct transform
        temp_model_name = "Salesforce/blip-image-captioning-base" # Or use a cfg variable
        temp_processor = BlipProcessor.from_pretrained(temp_model_name)
        image_transform = temp_processor.image_processor # Get the transform used by BLIP
        # --- MODIFICATION END ---

        dataset = ImageNet(cfg.DATASET.ROOT,
                           shift,
                           'val',
                           # transform=loader.dataset.transform # Use BLIP's transform instead
                           transform=image_transform # Apply BLIP's transform
                           )
        if args.attack == 'aa' and len(dataset) > 5000: # Limit AA dataset size if large
            dataset = torch.utils.data.Subset(dataset, list(range(5000)))
        loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=cfg.DATALOADER.TEST.BATCH_SIZE, # Use cfg batch size
                                             shuffle=False,
                                             num_workers=cfg.DATALOADER.NUM_WORKERS, # Use cfg workers
                                             pin_memory=True)

    # --- MODIFICATION START: Load Hugging Face BLIP Model and Processor ---
    # Remove old model loading and adversarial weight loading for standard BLIP ZS test
    # pretrained = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth"
    # model = blip_feature_extractor()
    # ckp_name = 'vitb32' if cfg.MODEL.BACKBONE.NAME == 'ViT-B/32' else 'rn50'
    # eps = int(cfg.AT.EPS * 255)
    # ckp_name += f'_eps{eps}.pth.tar'
    # ckp = torch.load(os.path.join('backbone', ckp_name))
    # model.visual_encoder.load_state_dict(ckp['vision_encoder_state_dict']) # Removed adversarial loading

    # Load HF BLIP
    model_name = "Salesforce/blip-image-captioning-base" # Consider making this configurable
    processor = BlipProcessor.from_pretrained(model_name)
    hf_blip_model = BlipModel.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hf_blip_model.to(device) # Move the core model to device
    # --- MODIFICATION END ---


    if 'prompter' in (args.cls_prompt, args.atk_prompt):
        # This prompter logic might need adjustments depending on how it interacts
        # with the zero-shot template format. Assumed to return a path or template string.
        prompter_path = os.path.join(cfg.OUTPUT_DIR, 'prompt_learner/')

        assert os.path.isdir(prompter_path), f"Prompter directory not found: {prompter_path}"
        if args.best_checkpoint:
            prompter_checkpoint_path = os.path.join(prompter_path, 'best.pth.tar')
        else:
            # Find the latest checkpoint if not 'best'
            ckp_files = [fname for fname in os.listdir(prompter_path) if 'model.pth.tar' in fname]
            if not ckp_files:
                raise FileNotFoundError(f"No 'model.pth.tar' checkpoint found in {prompter_path}")
            # Assuming checkpoints might have epoch numbers, sort to get the latest one if needed
            # Simple approach: take the first one found or implement sorting logic
            prompter_checkpoint_path = os.path.join(prompter_path, ckp_files[0]) # Adjust if specific checkpoint needed

        # How the prompter path/content is used needs clarification.
        # If it loads a learned prompter model, it needs integration.
        # If it's just a template string path, load it.
        # For now, assuming it provides the template string:
        if args.cls_prompt == 'prompter':
             # Assuming the checkpoint contains or points to the prompt template somehow
             print(f"WARNING: Loading actual prompter logic from {prompter_checkpoint_path} not implemented. Using path as placeholder.")
             # classify_prompt = load_prompt_template_from_path(prompter_checkpoint_path) # Placeholder function
             classify_prompt = args.cls_prompt # Fallback or adjust based on prompter implementation
        else:
             classify_prompt = args.cls_prompt

        # attack_prompt = prompter_checkpoint_path if args.atk_prompt == 'prompter' else args.atk_prompt
        # Attack prompt usage is unclear in ZS context, using default None or cls_prompt might be safer
        attack_prompt = None # Defaulting attack prompt to None for ZS
    else:
        classify_prompt = args.cls_prompt
        attack_prompt = args.atk_prompt # Keep original logic if not 'prompter'


    if args.linear_probe:
        # --- MODIFICATION START: Linear Probe requires adaptation ---
        print("WARNING: Linear Probe path needs specific adaptation for Hugging Face BLIP features. Skipping for now.")
        # Needs to:
        # 1. Define how to extract features from hf_blip_model (e.g., image features, multimodal features?)
        # 2. Adapt the LinearProbe class or create a new one to work with hf_blip_model.
        # 3. Load the linear layer weights correctly.
        # Example placeholder:
        # feature_dim = hf_blip_model.config.vision_config.hidden_size # Example: Use vision feature dimension
        # from adv_lp import LinearProbe # Assuming this class exists and can be adapted
        # model = LinearProbe(hf_blip_model, feature_dim, num_classes, False) # Needs adaptation
        # ckp_path = os.path.join(cfg.OUTPUT_DIR, 'linear_probe/linear.pth.tar')
        # if os.path.exists(ckp_path):
        #    ckp = torch.load(ckp_path)
        #    model.linear.load_state_dict(ckp) # Check if state dict keys match
        # else:
        #    print(f"Linear probe checkpoint not found at {ckp_path}")
        # model = model.to(device) # Move the final probe model to device
        exit("Linear Probe adaptation required. Exiting.") # Stop execution for LP for now
        # --- MODIFICATION END ---
    else:
        # --- MODIFICATION START: Instantiate the CustomBLIP wrapper ---
        model = CustomBLIP(processor,
                           hf_blip_model,
                           classes,
                           cls_prompt_template=classify_prompt) # Pass HF processor, model, classes, prompt
        # --- MODIFICATION END ---

    # No need to move model to cuda again if CustomBLIP handles device internally or hf_blip_model was moved
    # model = model.cuda()
    model.eval() # Set to evaluation mode

    meters = Dict()
    meters.acc = AverageMeter('Clean Acc@1', ':6.2f')
    meters.rob = AverageMeter('Robust Acc@1', ':6.2f')

    progress = ProgressMeter(
        len(loader),
        [meters.acc, meters.rob],
        prefix=f"{cfg.DATASET.NAME} ({args.dataset if args.dataset else 'Train Dist'})") # Add dataset info

    eps = cfg.AT.EPS # e.g., 8/255
    # --- MODIFICATION START: Ensure alpha calculation is float ---
    alpha = float(eps) / 4.0 # PGD step size
    # --- MODIFICATION END ---
    steps = 10 # Reduced steps for faster testing, increase for stronger attacks (e.g., 20, 40, 100)

    # --- MODIFICATION START: Pass the CustomBLIP model wrapper to attack libraries ---
    if args.attack == 'aa':
        attack = AutoAttack(model, # Pass the wrapper
                            norm='Linf',
                            eps=eps,
                            version='standard', # 'standard', 'plus', 'rand'
                            verbose=False,
                            device=device) # Specify device for AA
    elif args.attack == 'pgd':
        attack = PGD(model, # Pass the wrapper
                     eps=eps, alpha=alpha, steps=steps, random_start=True)
    elif args.attack == 'tpgd':
        attack = TPGD(model, # Pass the wrapper
                      eps=eps, alpha=alpha, steps=steps)
    # --- MODIFICATION END ---

    print(f"Starting evaluation on {cfg.DATASET.NAME} ({args.dataset if args.dataset else 'Train Dist'})")
    print(f"Attack: {args.attack}, Eps: {eps}, Steps: {steps}")
    print(f"Batch size: {loader.batch_size}, Num workers: {loader.num_workers}")
    print(f"Classifier: {'Linear Probe' if args.linear_probe else 'Zero-Shot BLIP'}")
    if not args.linear_probe:
        print(f"Zero-shot prompt template: '{classify_prompt}'")


    for i, data in enumerate(loader, start=1):
        try:
            # Few-shot data loader from Dassl might have dict format
            imgs, tgts = data['img'], data['label']
        except TypeError: # Handle cases where loader returns tuple/list
            imgs, tgts = data[0], data[1] # Assuming (image, target) format

        imgs, tgts = imgs.to(device), tgts.to(device)
        bs = imgs.size(0)

        # --- Clean Accuracy ---
        with torch.no_grad():
             output = model(imgs) # Uses CustomBLIP.forward for ZS logits

        acc = accuracy(output, tgts) # Assumes accuracy function takes logits
        meters.acc.update(acc[0].item(), bs)

        # --- Robust Accuracy ---
        # Removed mode switching: model.mode = 'attack' / 'classification'
        if args.attack == 'aa':
            # AutoAttack handles the evaluation loop internally for standard eval
            # It might be better to run AA outside the loop for the whole dataset?
            # Or run per batch like this:
            adv = attack.run_standard_evaluation(imgs, tgts, bs=bs if bs <= 100 else 100) # AA internal bs limit?
        elif args.attack in ['pgd', 'tpgd']:
            adv = attack(imgs, tgts) # PGD/TPGD directly return adversarial images
        elif args.attack == 'cw': # Example if using CWLoss with custom PGD
             # Use the custom pgd function with CWLoss
             # Note: CWLoss needs logits, which CustomBLIP provides
             adv, _ = pgd(imgs, tgts, model, CWLoss, eps, alpha, steps)
        else:
             print(f"Warning: Attack method '{args.attack}' not fully implemented or recognized. Skipping attack.")
             adv = imgs # Default to clean images if attack is unknown

        # Calculate accuracy on adversarial examples
        with torch.no_grad():
            output_adv = model(adv) # Get ZS logits for adversarial images

        rob = accuracy(output_adv, tgts)
        meters.rob.update(rob[0].item(), bs)

        if i == 1 or i % 10 == 0 or i == len(loader):
            progress.display(i)

    print(f"--- Evaluation Summary for {cfg.DATASET.NAME} ({args.dataset if args.dataset else 'Train Dist'}) ---")
    print(f"Clean Accuracy: {meters.acc.avg:.4f}")
    print(f"Robust Accuracy ({args.attack.upper()}): {meters.rob.avg:.4f}")

    # --- MODIFICATION START: Save results under appropriate keys ---
    # Determine the key for the dataset being evaluated
    current_eval_dataset = args.dataset if args.dataset else train_dataset
    if current_eval_dataset not in result:
        result[current_eval_dataset] = Dict()

    # Determine the key for the tuning method (ZS or LP)
    tune_key = 'linear_probe' if args.linear_probe else 'zero_shot'
    if tune_key not in result[current_eval_dataset]:
         result[current_eval_dataset][tune_key] = Dict()

    # Save clean and robust accuracy
    result[current_eval_dataset][tune_key].clean = meters.acc.avg
    result[current_eval_dataset][tune_key][args.attack] = meters.rob.avg

    # Ensure parent directory exists before saving
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # --- MODIFICATION END ---

    with open(save_path, 'w+') as f:
        # --- MODIFICATION START: Use sort_keys=False for better readability if needed ---
        yaml.dump(result.to_dict(), f, default_flow_style=False, sort_keys=False)
        # --- MODIFICATION END ---

    print(f'Result saved at: {save_path}')