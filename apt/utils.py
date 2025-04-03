import os
import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Tuple, TypeVar
from torch import Tensor

from clip import clip
from trainers.apt import PromptLearner, TextEncoder
from clip.simple_tokenizer import SimpleTokenizer 
from evaluate import load_clip_to_cpu
mu = (0.48145466, 0.4578275, 0.40821073)
std = (0.26862954, 0.26130258, 0.27577711)

class ImageNormalizer(nn.Module):

    def __init__(self, mean: Tuple[float, float, float],
                 std: Tuple[float, float, float]) -> None:
        super(ImageNormalizer, self).__init__()

        self.register_buffer('mean', torch.as_tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.as_tensor(std).view(1, 3, 1, 1))

    def forward(self, input: Tensor) -> Tensor:
        return (input - self.mean) / self.std

    def __repr__(self):
        return f'ImageNormalizer(mean={self.mean.squeeze()}, std={self.std.squeeze()})'  # type: ignore


class CustomCLIP(nn.Module):
    def __init__(self,
                 model,
                 classnames,
                 cls_prompt='a photo of a {}',
                 atk_prompt=None,
                 cfg=None):
        super().__init__()

        self.cfg = cfg
        self.logit_scale = model.logit_scale
        self.classnames = classnames
        self.model = model
        self.mode = 'classification'
        self.normalize = ImageNormalizer(mu, std).cuda()

        self.cls_prompt = cls_prompt 
        self.atk_prompt = atk_prompt
        
        self.set_prompts(cls_prompt, atk_prompt)
        
    def _prompt_text_features(self, prompt):
        if '{}' in prompt:
            # manual prompt template
            prompts = torch.cat([clip.tokenize(prompt.format(c))
                                 for c in self.classnames])
            self.model = self.model
            text_features = self.model.encode_text(prompts)
        else:
            # optimized prompt vector
            prompter_ckp = prompt
            assert os.path.isfile(prompter_ckp)
            prompter = PromptLearner(self.cfg, self.classnames, self.model)
            
            state_dict = torch.load(prompter_ckp)["state_dict"]
            
            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            prompter.load_state_dict(state_dict, strict=False)
            text_encoder = TextEncoder(self.model)
            prompts = prompter()
            text_features = text_encoder(prompts, prompter.tokenized_prompts)
            
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.detach(), prompts
        
    def set_prompts(self, cls_prompt, atk_prompt=None):
        print(f'classification prompt: {cls_prompt}')
        cls_tfeatures, cls_prompts = self._prompt_text_features(cls_prompt)
        self.cls_tfeatures = cls_tfeatures.cuda()
        self.cls_prompt = cls_prompts

        if atk_prompt is None or cls_prompt == atk_prompt:
            print(f'attack prompt: {cls_prompt}')
            self.atk_tfeatures = self.cls_tfeatures
            self.atk_prompt = self.cls_prompt
        else:
            print(f'attack prompt: {atk_prompt}')
            atk_tfeatures, atk_prompts = self._prompt_text_features(atk_prompt)
            self.atk_tfeatures = atk_tfeatures.cuda()
            self.atk_prompt = atk_prompts
                
    def forward(self, image):
        image_features = self.model.encode_image(self.normalize(image))        
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()

        text_features = self.cls_tfeatures if self.mode == 'classification' else self.atk_tfeatures
        logits = logit_scale * image_features @ text_features.t()
        
        return logits
    
    def _get_prompts(self):
        return {
            'classification_prompt': self.cls_prompt,
            'attack_prompt': self.atk_prompt
        }

def convert_to_raw(classify_prompt, classes, num_classes):
    prompt_learner_state = torch.load(classify_prompt, map_location='cpu')["state_dict"]
    ctx = prompt_learner_state["ctx"]
    ctx = ctx.float()
    print(f"Size of context: {ctx.shape}")

    tokenizer = SimpleTokenizer()
    clip_model = load_clip_to_cpu()
    token_embedding = clip_model.token_embedding.weight
    print(f"Size of token embedding: {token_embedding.shape}")

    if ctx.dim() == 2:
        # Generic context
        distance = torch.cdist(ctx, token_embedding)
        # print(f"Size of distance matrix: {distance.shape}")
        sorted_idxs = torch.argsort(distance, dim=1)
        sorted_idxs = sorted_idxs[:, :1]
        raw_words = []
        for m, idxs in enumerate(sorted_idxs):
            words = [tokenizer.decoder[idx.item()].replace('</w>', '') for idx in idxs]
            # print(f"Context {m+1}: {' '.join(words)}")
            raw_words.extend(words)
        raw_phrase = ' '.join(raw_words)
        class_raw_titles = [f"{raw_phrase} {classes[class_idx]}." for class_idx in range(num_classes)]
        return class_raw_titles
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

            sorted_idxs = torch.argsort(distance, dim=1)[:, :1]
            words_per_class = []
            for m, idxs in enumerate(sorted_idxs):
                words = [tokenizer.decoder[idx.item()].replace('</w>', '') for idx in idxs]
                # print(f"  Context token {m+1}: {' '.join(words)}")
                words_per_class.append(words[0])
            sentence = ' '.join(words_per_class)
            # print(f"Generated sentence for Class {class_idx + 1}: {sentence} class")
            class_raw_words.append(sentence)
        class_raw_titles = [f"{class_raw_words[class_idx]} {classes[class_idx]}" for class_idx in range(num_classes)]
        return class_raw_titles
    else:
        raise ValueError("Unsupported context dimension.")

class CustomBLIP(nn.Module):
    def __init__(self,
                 model,
                 processcor,
                 classnames,
                 cls_prompt='a photo of a {}',
                 atk_prompt=None,
                 cfg=None):
        super().__init__()

        self.cfg = cfg
        self.logit_scale = model.logit_scale
        self.classnames = classnames
        self.processor = processcor
        self.model = model
        self.mode = 'classification'
        self.normalize = ImageNormalizer(mu, std).cuda()

        self.cls_prompt = cls_prompt 
        self.atk_prompt = atk_prompt
        
        self.set_prompts(cls_prompt, atk_prompt)
        
    def _prompt_text_features(self, prompt):
        if '{}' in prompt:
            prompts_list = [prompt.format(c) for c in self.classnames]
            text_inputs = self.processor(text=prompts_list,
                                         return_tensors="pt",
                                         padding=True,
                                         truncation=True)
            text_inputs = {k: v for k, v in text_inputs.items()}

            with torch.no_grad():
                text_features = self.model.get_text_features(**text_inputs)
        else:
            prompts_list = convert_to_raw(prompt, self.classnames, len(self.classnames))
            print(prompts_list)
            text_inputs = self.processor(text=prompts_list,
                                         return_tensors="pt",
                                         padding=True,
                                         truncation=True)
            text_inputs = {k: v for k, v in text_inputs.items()}
            with torch.no_grad():
                text_features = self.model.get_text_features(**text_inputs)
            
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.detach(), text_inputs
        
    def set_prompts(self, cls_prompt, atk_prompt=None):
        print(f'classification prompt: {cls_prompt}')
        cls_tfeatures, cls_prompts = self._prompt_text_features(cls_prompt)
        self.cls_tfeatures = cls_tfeatures.cuda()
        self.cls_prompt = cls_prompts

        if atk_prompt is None or cls_prompt == atk_prompt:
            print(f'attack prompt: {cls_prompt}')
            self.atk_tfeatures = self.cls_tfeatures
            self.atk_prompt = self.cls_prompt
        else:
            print(f'attack prompt: {atk_prompt}')
            atk_tfeatures, atk_prompts = self._prompt_text_features(atk_prompt)
            self.atk_tfeatures = atk_tfeatures.cuda()
            self.atk_prompt = atk_prompts
                
    def forward(self, image):
        inputs = self.processor(images=image, return_tensors="pt", padding=True)
        pixel_values = inputs.pixel_values.cuda()
        image_features = self.model.get_image_features(pixel_values=pixel_values)      
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()

        text_features = self.cls_tfeatures if self.mode == 'classification' else self.atk_tfeatures
        logits = logit_scale * image_features @ text_features.t()
        
        return logits
    
    def _get_prompts(self):
        return {
            'classification_prompt': self.cls_prompt,
            'attack_prompt': self.atk_prompt
        }

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
