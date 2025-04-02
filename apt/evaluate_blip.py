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

# Thay thế CLIP imports bằng transformers
# import clip
from transformers import BlipProcessor, BlipForImageTextRetrieval # Hoặc mô hình BLIP phù hợp khác

import torch.nn as nn
from torch.autograd import grad, Variable
from torchvision import transforms
from torchvision.datasets import *
from collections import OrderedDict
from typing import Tuple, TypeVar, List
from addict import Dict
from dassl.data import DataManager

# Giữ lại các import dataset và utils
from datasets import (
    oxford_pets, oxford_flowers, fgvc_aircraft, dtd, eurosat,
    stanford_cars, food101, sun397, caltech101, ucf101, imagenet
)
from torchattacks import PGD, TPGD
from autoattack import AutoAttack
from utils import *
# Loại bỏ SimpleTokenizer của CLIP
# from clip.simple_tokenizer import SimpleTokenizer

# --- Hàm CWLoss và các hàm tấn công giữ nguyên ---
def CWLoss(output, target, confidence=0):
    # ... (giữ nguyên)
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
    # ... (giữ nguyên) - model ở đây sẽ là wrapper
    output = model(imgs)
    loss = criterion(output, targets)
    # Đảm bảo ảnh đầu vào cho phép tính gradient
    imgs_var = imgs.detach().clone().requires_grad_(True)
    output_for_grad = model(imgs_var)
    loss_for_grad = criterion(output_for_grad, targets)
    ig = grad(loss_for_grad, imgs_var, retain_graph=False, create_graph=False)[0]
    return ig


def perturb(imgs, targets, model, criterion, eps, eps_step, pert=None, ig=None):
    # ... (giữ nguyên) - model ở đây sẽ là wrapper
    adv = imgs.detach().clone().requires_grad_(True) if pert is None else torch.clamp(imgs+pert, 0, 1).requires_grad_(True)
    ig = input_grad(adv, targets, model, criterion) if ig is None else ig
    if pert is None:
        pert = eps_step*torch.sign(ig)
    else:
        pert = pert.detach() + eps_step*torch.sign(ig) # Ensure pert is detached before addition
    pert = torch.clamp(pert, -eps, eps)
    adv_perturbed = torch.clamp(imgs.detach() + pert, 0, 1) # Ensure imgs is detached
    pert = adv_perturbed - imgs.detach()
    return adv_perturbed.detach(), pert.detach()


def pgd(imgs, targets, model, criterion, eps, eps_step, max_iter, pert=None, ig=None):
    # ... (giữ nguyên) - model ở đây sẽ là wrapper
    imgs_pgd = imgs.detach().clone()
    pert_pgd = pert.detach().clone() if pert is not None else None
    for i in range(max_iter):
        adv, pert_pgd = perturb(imgs_pgd, targets, model, criterion, eps, eps_step, pert_pgd, ig)
        imgs_pgd = adv # Update image for next iteration
        ig = None # Recalculate gradient in the next perturb step
    return imgs_pgd, pert_pgd

# Loại bỏ hàm load_clip_to_cpu
# def load_clip_to_cpu(backbone_name="RN50"): ...

# --- Định nghĩa Lớp Wrapper cho BLIP ---
class BlipZeroShotWrapper(nn.Module):
    """
    Wrapper class for BLIP model to make it compatible with attack libraries
    that expect a standard forward(images) -> logits interface.
    """
    def __init__(self, blip_model, blip_processor, text_prompts: List[str], device):
        super().__init__()
        self.blip_model = blip_model
        self.blip_processor = blip_processor
        self.text_prompts = text_prompts
        self.device = device
        # Đảm bảo mô hình BLIP ở chế độ eval
        self.blip_model.eval()

    def forward(self, images):
        # images: [batch_size, channels, height, width]
        batch_size = images.shape[0]
        num_prompts = len(self.text_prompts)
        all_scores = torch.zeros(batch_size, num_prompts).to(self.device)

        # Tính điểm ITM cho từng prompt
        # Không cần torch.no_grad() ở đây vì wrapper này có thể được dùng
        # bởi các thư viện tấn công cần gradient qua nó.
        # Tuy nhiên, bản thân model BLIP bên trong nên được giữ ở eval()
        # và các tính toán gradient cho ảnh sẽ được xử lý bởi thư viện tấn công.

        for i, prompt in enumerate(self.text_prompts):
            # Xử lý batch ảnh với một prompt
            # Lặp lại prompt cho cả batch
            prompts_batch = [prompt] * batch_size
            inputs = self.blip_processor(
                images=images,
                text=prompts_batch,
                return_tensors="pt",
                padding="max_length", # Hoặc True, đảm bảo padding nhất quán
                truncation=True,
                max_length=77 # Hoặc một độ dài phù hợp cho BLIP
            ).to(self.device)

            # Tính toán điểm ITM
            outputs = self.blip_model(**inputs)
            # outputs.logits[:, 1] thường là điểm số "khớp" (match) cho ITM
            itm_scores = outputs.logits[:, 1]
            all_scores[:, i] = itm_scores

        # all_scores bây giờ có shape [batch_size, num_classes]
        return all_scores


# --- Parser và Main Script ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # --- Giữ nguyên các argument parser ---
    parser.add_argument('experiment', type=str, help="Name or type of experiment to run.")
    parser.add_argument('-cp', '--cls-prompt', type=str, default='a photo of a {}',
                        help="Template for class prompt. Default is 'a photo of a {}'.")
    # parser.add_argument('-ap', '--atk-prompt', type=str, default=None, # Attack prompt không còn ý nghĩa trực tiếp với cách tiếp cận ITM này
    #                     help="Template for attack prompt. If not specified, defaults to None.")
    parser.add_argument('--best-checkpoint', action='store_true',
                        help="Use the best checkpoint if available (Not directly applicable for standard BLIP loading).")
    parser.add_argument('--attack', type=str, default='pgd', choices=['pgd', 'tpgd', 'aa', 'cw'], # Thêm 'cw' nếu muốn
                        help="Type of attack to use. Default is 'pgd'.")
    parser.add_argument('--dataset', type=str, default=None,
                        help="Dataset to use for the experiment. Defaults to None.")
    parser.add_argument('-lp', '--linear-probe', action='store_true',
                        help="Enable linear probing for the experiment (Currently NOT fully supported with BLIP integration).")
    parser.add_argument('--save-img', action='store_true',
                        help="Enable save images for the experiment.")
    parser.add_argument('--save-path', type=str, default = './',
                        help="Specific path to save images. Default is ./")
    parser.add_argument('--num-imgs', type=int, default = 10,
                        help="Number of images to save. Default is 10")
    parser.add_argument('--seed', type=int, default = 42,
                        help="Seed for torch random")
    # parser.add_argument("--topk", type=int, default = '1', # Không còn dùng với text prompt cố định
    #                     help="Select top-k similar words (Not used with fixed prompts)")
    parser.add_argument('--blip-model-name', type=str, default='Salesforce/blip-itm-base-coco',
                        help="Name of the BLIP model to load from Hugging Face.")

    args = parser.parse_args()

    if args.linear_probe:
        warn("Linear probing is not fully integrated with BLIP in this script yet. Disabling.")
        args.linear_probe = False # Tạm thời vô hiệu hóa

    if args.cls_prompt == 'prompter':
         warn("Learnable prompts ('prompter') are not supported with this BLIP integration. Using fixed prompts based on --cls-prompt template.")
         args.cls_prompt = 'a photo of a {}' # Hoặc giá trị mặc định khác

    cfg = CfgNode()
    cfg.set_new_allowed(True)
    # Kiểm tra xem args.experiment có phải là đường dẫn file hay không
    if os.path.isfile(args.experiment):
         cfg_path = args.experiment
         # Lấy tên thư mục chứa file config làm tên experiment nếu cần
         exp_name_from_path = os.path.basename(os.path.dirname(args.experiment))
         # Tạo đường dẫn lưu kết quả dựa trên thư mục của config
         default_save_path = os.path.join(os.path.dirname(cfg_path), 'results') # Lưu vào thư mục con 'results'
         print(f"Config path is a file: {cfg_path}")
    else:
         # Giả định args.experiment là tên thư mục chứa cfg.yaml
         cfg_path = os.path.join(args.experiment, 'cfg.yaml')
         default_save_path = args.experiment # Lưu vào chính thư mục experiment
         print(f"Config path is a directory, looking for cfg.yaml: {cfg_path}")

    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Configuration file not found at {cfg_path}")

    cfg.merge_from_file(cfg_path)

    # Xác định save_path cuối cùng
    output_save_path = args.save_path if args.save_path != './' else default_save_path
    os.makedirs(output_save_path, exist_ok=True)
    print(f"Results will be saved in: {output_save_path}")


    train_dataset = cfg.DATASET.NAME
    # Cập nhật save_path dựa trên output_save_path đã xác định
    if args.dataset:
        if args.dataset in ['ImageNetR', 'ImageNetA', 'ON']:
            cfg.DATASET.NAME = 'ImageNet' # Giữ nguyên logic xử lý dataset ImageNet đặc biệt
        else:
            cfg.DATASET.NAME = args.dataset
        result_file = os.path.join(output_save_path, f'dist_shift_{args.dataset}.yaml')
    else:
        result_file = os.path.join(output_save_path, 'evaluation.yaml')

    # --- Logic kiểm tra file kết quả đã tồn tại ---
    result = Dict()
    if os.path.isfile(result_file):
        try:
            with open(result_file, 'r') as f:
                result = Dict(yaml.safe_load(f) or {}) # Đảm bảo là Dict ngay cả khi file trống
        except yaml.YAMLError:
            print(f"Warning: Could not parse existing results file: {result_file}. Starting fresh.")
            result = Dict() # Bắt đầu lại nếu file bị lỗi

    result_key = args.dataset if args.dataset and args.dataset != train_dataset else 'default'
    tune_key = 'linear_probe' if args.linear_probe else args.cls_prompt
    attack_key = args.attack

    # Khởi tạo cấu trúc nếu chưa có
    if result_key not in result:
        result[result_key] = Dict()
    if tune_key not in result[result_key]:
        result[result_key][tune_key] = Dict()
    if attack_key not in result[result_key][tune_key]:
         result[result_key][tune_key][attack_key] = {} # Dùng dict trống thay vì None để kiểm tra dễ hơn

    # Kiểm tra xem kết quả cụ thể đã tồn tại chưa
    if result[result_key][tune_key][attack_key] != {}:
          print(f"Evaluation result already exists in {result_file} for Key: {result_key}, Tune: {tune_key}, Attack: {attack_key}. Skipping.")
          # exit() # Bỏ exit để có thể chạy lại nếu muốn ghi đè

    # --- Tải dữ liệu (giữ nguyên DataManager) ---
    print("Loading dataset...")
    # Có thể cần điều chỉnh transform trong DataManager nếu nó quá đặc thù cho CLIP
    # Ví dụ: đảm bảo nó trả về PIL Image hoặc Tensor cơ bản
    dm = DataManager(cfg)
    classes = dm.dataset.classnames
    # Sử dụng test_loader cho đánh giá
    try:
        loader = dm.test_loader
        print(f"Using test_loader. Batch size: {loader.batch_size}")
    except AttributeError:
        print("Warning: test_loader not found in DataManager. Falling back to val_loader.")
        loader = dm.val_loader # Dự phòng nếu test_loader không có
        print(f"Using val_loader. Batch size: {loader.batch_size}")

    if loader is None:
         raise ValueError("Could not find a suitable data loader (test_loader or val_loader).")

    num_classes = dm.num_classes
    print(f"Number of classes: {num_classes}")

    # --- Xử lý các dataset đặc biệt (ImageNetR/A/V2/ON) ---
    if args.dataset in ['ImageNetV2','ImageNetR', 'ImageNetA', 'ON'] or (train_dataset == 'ImageNet' and args.dataset is None and args.attack == 'aa'):
        # ... (giữ nguyên logic tải dataset ImageNet đặc biệt) ...
        print(f"Loading special dataset: {args.dataset or 'ImageNet for AA'}")
        # Note: Ensure the transform used here is compatible with BLIP's processor later
        # Usually, a basic ToTensor() and Resize() might suffice, letting the processor handle normalization.
        # Let's assume the original transform is somewhat generic or adaptable.
        # If issues arise, this transform might need adjustment.
        basic_transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224), # Kích thước chuẩn cho nhiều mô hình Vision
            transforms.ToTensor(),
            # Normalization sẽ được thực hiện bởi BlipProcessor sau
        ])

        from OODRB.imagenet import ImageNet # Đảm bảo import này đúng
        if args.dataset == 'ImageNetV2': shift = 'v2'
        elif args.dataset == 'ImageNetA': shift = 'A'
        elif args.dataset == 'ImageNetR': shift = 'R'
        elif args.dataset == 'ON': shift = 'ON'
        else: shift = None # Cho trường hợp ImageNet thường + AA

        num_classes = 1000 # ImageNet luôn có 1000 lớp
        print("Loading ImageNet distribution shift dataset...")
        try:
            # Sử dụng transform cơ bản hơn
            dataset = ImageNet(cfg.DATASET.ROOT, shift, 'val', transform=basic_transform)
            print(f"Loaded {len(dataset)} images for ImageNet {shift or '(for AA)'}")
            if args.attack == 'aa' and shift is None: # Chỉ subset nếu là ImageNet thường cho AA
                subset_indices = list(range(min(5000, len(dataset))))
                dataset = torch.utils.data.Subset(dataset, subset_indices)
                print(f"Subsetting to {len(dataset)} images for AutoAttack.")

            loader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=cfg.DATALOADER.TEST.BATCH_SIZE, # Sử dụng batch size từ config
                                                 shuffle=False,
                                                 num_workers=cfg.DATALOADER.NUM_WORKERS,
                                                 pin_memory=True)
            print(f"Created DataLoader for special dataset. Batch size: {cfg.DATALOADER.TEST.BATCH_SIZE}")
        except Exception as e:
            print(f"Error loading ImageNet distribution shift dataset: {e}")
            raise


    # --- Tải mô hình BLIP ---
    print(f"Loading BLIP model: {args.blip_model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        processor = BlipProcessor.from_pretrained(args.blip_model_name)
        blip_model = BlipForImageTextRetrieval.from_pretrained(args.blip_model_name)
        blip_model.to(device)
        blip_model.eval() # Luôn đặt ở chế độ đánh giá
    except Exception as e:
        print(f"Error loading BLIP model '{args.blip_model_name}': {e}")
        print("Please ensure the model name is correct and you have internet connection.")
        print("Common ITM models: Salesforce/blip-itm-base-coco, Salesforce/blip-itm-large-coco")
        print("Common VQA models (less ideal for ZS classification): Salesforce/blip-vqa-base")
        raise

    # --- Bỏ qua việc tải backbone mạnh mẽ riêng và prompter ---
    # ckp_name = ...
    # ckp = torch.load(...)
    # model.visual.load_state_dict(...)
    # prompter_path = ...

    # --- Tạo Text Prompts ---
    print("Generating text prompts...")
    if args.linear_probe:
        # Logic Linear Probe sẽ cần viết lại hoàn toàn cho BLIP features
        print("Linear Probe is selected but not implemented for BLIP. Exiting.")
        exit() # Hoặc thực hiện logic thay thế nếu có
    else:
        # Sử dụng template từ args.cls_prompt
        try:
             text_prompts = [args.cls_prompt.format(c) for c in classes]
             print(f"Generated {len(text_prompts)} prompts using template: '{args.cls_prompt}'")
             # print("Example prompts:", text_prompts[:3])
        except Exception as e:
             print(f"Error formatting prompts with template '{args.cls_prompt}' and classes: {e}")
             print("Ensure the template has a placeholder like {}")
             raise

    # --- Tạo Wrapper Model cho BLIP ---
    # Wrapper này sẽ được dùng cho cả đánh giá clean và adversarial
    model = BlipZeroShotWrapper(blip_model, processor, text_prompts, device)
    model.eval() # Đảm bảo wrapper cũng ở chế độ eval (dù lớp bên trong đã eval)

    # --- Logic lấy prompt học được đã bị loại bỏ ---
    # if args.cls_prompt == 'prompter': ...

    # --- Khởi tạo Meters ---
    meters = Dict()
    meters.acc = AverageMeter('Clean Acc@1', ':6.2f')
    meters.rob = AverageMeter('Robust Acc@1', ':6.2f')

    progress = ProgressMeter(
        len(loader),
        [meters.acc, meters.rob],
        prefix=f"{cfg.DATASET.NAME} Eval")

    # --- Thiết lập Tấn công ---
    # eps nên lấy từ config nếu có, hoặc đặt giá trị mặc định
    eps = cfg.AT.EPS if hasattr(cfg, 'AT') and hasattr(cfg.AT, 'EPS') else 8/255
    # alpha nên tỉ lệ với eps
    alpha = eps / 4.0
    # steps nên lấy từ config nếu có, hoặc đặt giá trị mặc định
    steps = cfg.AT.STEPS if hasattr(cfg, 'AT') and hasattr(cfg.AT, 'STEPS') else 10 # Giảm steps để chạy nhanh hơn nếu cần test

    print(f"Attack Settings: Type={args.attack}, Eps={eps:.4f}, Alpha={alpha:.4f}, Steps={steps}")

    # Lưu ý: Truyền wrapper `model` vào các lớp tấn công
    if args.attack == 'aa':
        # AutoAttack cần đường dẫn lưu log, hoặc tắt verbose
        log_path = os.path.join(output_save_path, f'aa_log_{args.dataset or train_dataset}.txt')
        print(f"AutoAttack log path: {log_path}")
        attack = AutoAttack(model, norm='Linf', eps=eps, version='standard', verbose=False, log_path=log_path)
        # attack.attacks_to_run = ['apgd-ce', 'apgd-t'] # Có thể giới hạn các attack con để nhanh hơn
    elif args.attack == 'pgd':
        attack = PGD(model, eps=eps, alpha=alpha, steps=steps, random_start=True)
    elif args.attack == 'tpgd':
        attack = TPGD(model, eps=eps, alpha=alpha, steps=steps) # TPGD thường dùng loss khác, nhưng ở đây dùng CE mặc định của torchattacks
    elif args.attack == 'cw':
        # CW attack trong torchattacks thường dùng targeted.
        # Để dùng untargeted, có thể cần custom hoặc dùng loss CW với PGD.
        # Ở đây ta dùng PGD với CW loss.
        print("Using PGD with CWLoss for 'cw' attack type.")
        attack = PGD(model, eps=eps, alpha=alpha, steps=steps, random_start=True)
        # Loss sẽ được truyền vào khi gọi attack sau
    else:
         raise ValueError(f"Unsupported attack type: {args.attack}")


    # --- Chuẩn bị lưu ảnh nếu cần ---
    if args.save_img:
        clean_dir = os.path.join(output_save_path, f'images_{args.dataset or train_dataset}_clean')
        adv_dir = os.path.join(output_save_path, f'images_{args.dataset or train_dataset}_adv_{args.attack}')
        os.makedirs(clean_dir, exist_ok=True)
        os.makedirs(adv_dir, exist_ok=True)
        print(f'Saving clean images in: {clean_dir}')
        print(f'Saving adversarial images in: {adv_dir}')

        all_logits_clean = []
        all_images_clean = []
        all_logits_adv = []
        all_images_adv = []
        all_labels = []
        saved_img_count = 0 # Đếm số lượng ảnh đã lưu

    # --- Vòng lặp Đánh giá ---
    print("Starting evaluation loop...")
    for i, data in enumerate(tqdm(loader, desc=f"Evaluating {cfg.DATASET.NAME}"), start=1):
        try:
            # Xử lý cả hai kiểu loader (từ Dassl hoặc PyTorch chuẩn)
            if isinstance(data, dict) and 'img' in data and 'label' in data:
                 imgs, tgts = data['img'], data['label']
            elif isinstance(data, (list, tuple)) and len(data) >= 2:
                 imgs, tgts = data[0], data[1]
            else:
                 print(f"Warning: Unexpected data format at batch {i}. Skipping.")
                 continue
        except Exception as e:
             print(f"Error unpacking data at batch {i}: {e}. Skipping batch.")
             continue

        # Đảm bảo dữ liệu ở đúng device
        # Processor sẽ xử lý ảnh PIL hoặc Tensor, nên chuyển lên GPU sau khi xử lý nếu cần
        # Tuy nhiên, các lớp attack thường mong đợi Tensor trên GPU
        imgs, tgts = imgs.to(device), tgts.to(device)
        bs = imgs.size(0)

        if bs == 0:
            print(f"Warning: Empty batch at index {i}. Skipping.")
            continue

        # --- Đánh giá Clean ---
        with torch.no_grad():
            # Sử dụng wrapper model
            output_clean = model(imgs)

        acc = accuracy(output_clean, tgts) # Giả sử accuracy trả về list/tuple [acc1, acc5]
        meters.acc.update(acc[0].item(), bs)

        # --- Tạo Ảnh Adversarial ---
        # Không cần đặt model.mode vì wrapper luôn hoạt động theo một cách
        if args.attack == 'aa':
             # AutoAttack xử lý batching nội bộ nếu cần
             # Nó mong đợi tgts trên CPU cho một số logging nội bộ, nhưng model nhận tgts trên device
             adv = attack.run_standard_evaluation(imgs, tgts, bs=loader.batch_size) # Cung cấp bs để AA biết cách chia batch nếu cần
        elif args.attack in ['pgd', 'tpgd']:
             adv = attack(imgs, tgts)
        elif args.attack == 'cw':
             # Sử dụng PGD với CWLoss
             # Lưu ý: CWLoss cần logits, không phải probs
             adv = attack(imgs, lambda m, i, t: CWLoss(m(i), t)) # Truyền loss function vào attack
        else: # cw PGD-style
             # Logic pgd thủ công đã bị loại bỏ để dùng torchattacks
             adv, _ = pgd(imgs, tgts, model, CWLoss, eps, alpha, steps)


        # --- Đánh giá Robust ---
        with torch.no_grad():
             output_adv = model(adv)

        rob = accuracy(output_adv, tgts)
        meters.rob.update(rob[0].item(), bs)

        # --- Lưu ảnh (nếu cần và chưa đủ số lượng) ---
        if args.save_img and saved_img_count < args.num_imgs:
             num_to_save = min(args.num_imgs - saved_img_count, bs)
             if num_to_save > 0:
                 # Lưu trữ batch hiện tại để xử lý sau vòng lặp (hiệu quả hơn)
                 all_labels.append(tgts[:num_to_save].cpu())
                 all_logits_clean.append(output_clean[:num_to_save].cpu())
                 # Chuyển ảnh về CPU và định dạng phù hợp (ví dụ: float 0-1)
                 all_images_clean.append(imgs[:num_to_save].cpu().float())
                 all_logits_adv.append(output_adv[:num_to_save].cpu())
                 all_images_adv.append(adv[:num_to_save].cpu().float())
                 saved_img_count += num_to_save


        # In tiến trình
        # if i == 1 or i % 10 == 0 or i == len(loader): # In thường xuyên hơn
        #     progress.display(i)
        # Cập nhật tqdm description thay vì in thủ công
        progress.batch_fmt = progress.batch_fmtstr(i, len(loader)) # Cập nhật số batch hiện tại
        tqdm_desc = progress.get_message(i)
        loader.set_description_str(tqdm_desc)


    # In kết quả cuối cùng
    print("\n--- Evaluation Summary ---")
    print(f"Dataset: {cfg.DATASET.NAME}")
    print(f"BLIP Model: {args.blip_model_name}")
    print(f"Prompt Template: '{args.cls_prompt}'")
    print(f"Attack: {args.attack} (eps={eps:.4f})")
    print(f"* Clean Accuracy@1: {meters.acc.avg:.2f}%")
    print(f"* Robust Accuracy@1: {meters.rob.avg:.2f}%")
    print("-------------------------")


    # --- Xử lý và Lưu Ảnh đã thu thập ---
    if args.save_img and all_labels: # Chỉ xử lý nếu có ảnh để lưu
        print(f"\nProcessing and saving {saved_img_count} images...")
        all_logits_clean = torch.cat(all_logits_clean, dim=0)
        all_images_clean = torch.cat(all_images_clean, dim=0)
        all_logits_adv = torch.cat(all_logits_adv, dim=0)
        all_images_adv = torch.cat(all_images_adv, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Đảm bảo ảnh nằm trong khoảng [0, 1] sau khi tấn công
        all_images_clean.clamp_(0, 1)
        all_images_adv.clamp_(0, 1)

        torch.manual_seed(args.seed) # Đặt seed để chọn ngẫu nhiên nhất quán (nếu cần)

        # Lặp qua từng lớp để lưu ảnh đại diện
        unique_labels = torch.unique(all_labels)
        for class_idx in tqdm(unique_labels, desc="Saving images per class"):
            class_idx_item = class_idx.item()
            if class_idx_item >= len(classes):
                 print(f"Warning: Label index {class_idx_item} out of bounds for classes list (size {len(classes)}). Skipping.")
                 continue
            class_name = classes[class_idx_item]

            indices_for_class = (all_labels == class_idx).nonzero(as_tuple=False).squeeze()
            # Xử lý trường hợp chỉ có 1 ảnh cho lớp đó
            if indices_for_class.numel() == 0:
                continue
            if indices_for_class.dim() == 0:
                 indices_for_class = indices_for_class.unsqueeze(0)

            # Lấy dữ liệu cho lớp hiện tại
            logits_cls_clean = all_logits_clean[indices_for_class]
            images_cls_clean = all_images_clean[indices_for_class]
            logits_cls_adv = all_logits_adv[indices_for_class]
            images_cls_adv = all_images_adv[indices_for_class]

            # Chọn ngẫu nhiên k ảnh từ lớp này
            k = min(5, images_cls_clean.size(0)) # Lưu tối đa 5 ảnh/lớp để tránh quá nhiều file
            if k == 0: continue
            random_indices = torch.randperm(images_cls_clean.size(0))[:k]

            selected_logits_clean = logits_cls_clean[random_indices]
            selected_images_clean = images_cls_clean[random_indices]
            selected_logits_adv = logits_cls_adv[random_indices]
            selected_images_adv = images_cls_adv[random_indices]

            # Tạo plot và lưu ảnh clean
            try:
                fig_clean, axes_clean = plt.subplots(1, k, figsize=(3*k, 4))
                if k == 1: axes_clean = [axes_clean] # Đảm bảo axes luôn là list
                # fig_clean.suptitle(f"Clean - Class: {class_name}", fontsize=10)
                for j, ax in enumerate(axes_clean):
                    img = np.transpose(selected_images_clean[j].numpy(), (1, 2, 0)) # Chuyển về HWC
                    ax.imshow(img)
                    ax.axis('off')
                    pred_idx_clean = selected_logits_clean.argmax(dim=1)[j].item()
                    pred_class_clean = classes[pred_idx_clean] if pred_idx_clean < len(classes) else "Unknown"
                    title = f"Pred: {pred_class_clean}"
                    if pred_idx_clean != class_idx_item:
                         title += f"\n(True: {class_name})" # Chỉ hiển thị True nếu dự đoán sai
                    ax.set_title(title, fontsize=8)
                plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
                safe_class_name = "".join(c if c.isalnum() else "_" for c in class_name) # Tên file an toàn
                plt.savefig(os.path.join(clean_dir, f'class_{safe_class_name}_clean.png'))
                plt.close(fig_clean)
            except Exception as e:
                 print(f"Error saving clean image plot for class {class_name}: {e}")
                 plt.close('all') # Đóng tất cả plot nếu có lỗi

            # Tạo plot và lưu ảnh adv
            try:
                fig_adv, axes_adv = plt.subplots(1, k, figsize=(3*k, 4))
                if k == 1: axes_adv = [axes_adv]
                # fig_adv.suptitle(f"Adv ({args.attack}) - Class: {class_name}", fontsize=10)
                for j, ax in enumerate(axes_adv):
                    img = np.transpose(selected_images_adv[j].numpy(), (1, 2, 0))
                    ax.imshow(img)
                    ax.axis('off')
                    pred_idx_adv = selected_logits_adv.argmax(dim=1)[j].item()
                    pred_class_adv = classes[pred_idx_adv] if pred_idx_adv < len(classes) else "Unknown"
                    title = f"Pred: {pred_class_adv}"
                    if pred_idx_adv != class_idx_item:
                         title += f"\n(True: {class_name})"
                    ax.set_title(title, fontsize=8)
                plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
                plt.savefig(os.path.join(adv_dir, f'class_{safe_class_name}_adv.png'))
                plt.close(fig_adv)
            except Exception as e:
                 print(f"Error saving adversarial image plot for class {class_name}: {e}")
                 plt.close('all') # Đóng tất cả plot nếu có lỗi

        print("Finished saving images.")


    # --- Lưu Kết quả vào File YAML ---
    print(f"\nSaving results to: {result_file}")
    # Ghi đè kết quả cho lần chạy này
    result[result_key][tune_key]['clean'] = meters.acc.avg
    result[result_key][tune_key][attack_key] = meters.rob.avg # Lưu giá trị avg trực tiếp

    try:
        with open(result_file, 'w+') as f:
            # Chuyển đổi Dict addict thành dict Python chuẩn trước khi dump
            yaml.dump(result.to_dict(), f, default_flow_style=False, sort_keys=False)
        print("Results saved successfully.")
    except Exception as e:
        print(f"Error saving results to YAML file: {e}")

    print("\nScript finished.")