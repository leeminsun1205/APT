import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from lavis.models import load_model_and_preprocess
from transformers import BertTokenizer

import sys

def load_blip_to_cpu(device="cpu"):
    # Load BLIP feature extractor
    model, val_processors, text_processors = load_model_and_preprocess(
        name="blip_feature_extractor", model_type="base", is_eval=True, device=device
    )
    return model, text_processors

class BLIPTextEncoder(nn.Module):
    def __init__(self, blip_model):
        super().__init__()
        self.text_encoder = blip_model.text_encoder
        self.tokenizer = blip_model.tokenizer # Usually this is available or we need to init one
        
    def forward(self, prompts, tokenized_prompts):
        # prompts: (n_cls, n_ctx, ctx_dim) or (n_b, n_ctx, ctx_dim)
        # tokenized_prompts: keys like input_ids, attention_mask
        
        # If prompts are passed as embeddings, we need to merge them with the token embeddings
        # BUT: this logic is typically handled inside CustomBLIP or outside TextEncoder in CLIP impl.
        # Here we just wrap the forward call to support inputs_embeds if possible.
        
        # For simplicity, let's assume this module just blindly forwards whatever it gets
        # matching standard HF BertModel signature
        return self.text_encoder(**tokenized_prompts)

class BLIPPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, blip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = torch.float32 # BLIP is fp32 by default usually, unlike CLIP's fp16
        ctx_dim = 768 # BERT-base dimension
        # clip_imsize = blip_model.visual_encoder.img_size # e.g. 384 or 224 - Not reliable


        if ctx_init:
            # Initialize with given words
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = ctx_init
            with torch.no_grad():
                # We need a tokenizer. Check if blip_model has one.
                # Assuming blip_model has .tokenizer which is a BertTokenizer
                tokenizer = blip_model.tokenizer
                tokenized = tokenizer(prompt, return_tensors="pt")
                # Get embeddings
                # blip_model.text_encoder is XBertEncoder
                # embeddings layer: text_encoder.embeddings.word_embeddings
                embedding = blip_model.text_encoder.embeddings.word_embeddings(tokenized.input_ids)
            ctx_vectors = embedding[0, 1:1+n_ctx, :] # Skip [CLS]
            prompt_prefix = ctx_init
        else:
            # Random initialization
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        self.classnames = classnames
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.ctx_dim = ctx_dim
        self.tokenizer = blip_model.tokenizer
        self.blip_model = blip_model 
        
        # Construct and cache class prompts (the suffix part)
        # For each class, prompt is "[CTX] [CTX] ... [CLASS]"
        # We need to construct the templates
        class_tokens = []
        for name in classnames:
            # Tokenize class name
            # We assume simple single-token or few-token class names for now
            # But we must handle tokenization properly
            name = name.replace("_", " ")
            tokenized = self.tokenizer(name, add_special_tokens=False) 
            # Note: We append this to Context. Context is injected manually.
            # We need the EMBEDDINGS of the class name.
            with torch.no_grad():
                ids = torch.tensor([tokenized["input_ids"]], device=blip_model.device)
                embeds = blip_model.text_encoder.embeddings.word_embeddings(ids)[0]
            class_tokens.append(embeds)
            
        # We store class name embeddings, not tokens, because we will concat embeddings
        self.register_buffer("class_embeddings_list", torch.cat([e.unsqueeze(0) for e in class_tokens])) # List of tensors size varies if class lengths vary?
        # Actually simplest is to pad class tokens or assume fixed length?
        # For simplicity, if lengths vary, we can't easily stack. 
        # But CoOp assumes simple templates.
        
        # Let's rely on constructing text and only learning the context vectors,
        # but injecting them is tricky if we do it at token level.
        # We must splice inputs_embeds.
        
        # Alternative: Pre-compute part of the prompt that is fixed?
        # To support variable length class names, we usually iterate or pad.
        # Let's save the token IDs and computing embeddings on the fly might be safer 
        # or pre-compute if we trust they don't change.
        
        # For batch processing, we need equal length.
        # We'll tokenize "a photo of a {}" ?
        # No, CoOp just uses "X X X classname".
        
        self.class_token_ids = [self.tokenizer(name.replace("_", " "), add_special_tokens=False)["input_ids"] for name in classnames]
        
    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        # Iterate over classes to build the full batch of embeddings
        # This is inefficient but BLIP text encoder might not handle batched custom embeddings easily without padding
        
        # We are building [CLS] [CTX]...[CTX] [CLASS] [SEP]
        
        # 1. Get [CLS] and [SEP] embeddings
        with torch.no_grad():
            cls_id = self.tokenizer.cls_token_id
            sep_id = self.tokenizer.sep_token_id
            cls_embed = self.blip_model.text_encoder.embeddings.word_embeddings(
                torch.tensor([[cls_id]], device=self.ctx.device)
            ) # (1, 1, dim)
            sep_embed = self.blip_model.text_encoder.embeddings.word_embeddings(
                torch.tensor([[sep_id]], device=self.ctx.device)
            )
            
        # Construct batch
        batch_embeds = []
        max_len = 0
        
        # We need to construct the full embeddings for each class
        for i, class_ids in enumerate(self.class_token_ids):
            # class_ids is list of ints
            ids_tensor = torch.tensor([class_ids], device=self.ctx.device)
            with torch.no_grad():
                class_embed = self.blip_model.text_encoder.embeddings.word_embeddings(ids_tensor)
            
            # Concat: [CLS] + CTX[i] + NAME + [SEP]
            # ctx[i]: (n_ctx, dim) -> (1, n_ctx, dim)
            ctx_i = ctx[i].unsqueeze(0)
            
            # Final: (1, 1+n_ctx+len_name+1, dim)
            full_embed = torch.cat([cls_embed, ctx_i, class_embed, sep_embed], dim=1)
            batch_embeds.append(full_embed)
            max_len = max(max_len, full_embed.shape[1])
            
        # Stack and pad
        # Or simpler: if all class names are 1 token (often true for CIFAR100), fast path.
        # padding value should be 0 (masked out). 
        # But we are dealing with Embeddings. We can pad with zeros.
        
        final_embeds = torch.zeros(self.n_cls, max_len, self.ctx_dim, device=self.ctx.device, dtype=ctx.dtype)
        attn_mask = torch.zeros(self.n_cls, max_len, device=self.ctx.device, dtype=torch.long)
        
        for i, emb in enumerate(batch_embeds):
            leng = emb.shape[1]
            final_embeds[i, :leng, :] = emb
            attn_mask[i, :leng] = 1
            
        return final_embeds, attn_mask

class CustomBLIP(nn.Module):
    def __init__(self, cfg, classnames, blip_model, device):
        super().__init__()
        self.prompt_learner = BLIPPromptLearner(cfg, classnames, blip_model)
        self.classnames = classnames
        self.blip_model = blip_model
        self.device = device # redundancy
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592) # CLIP init (np.log(1/0.07)) ? BLIP might use different scaling. 
        # BLIP image-text matching uses manual dot product often or ITM head. 
        # But for classification (zero-shot), it compares image features and text features (CLS token).
        # We will use dot product + learned scale similar to CLIP.
        
    def forward(self, image):
        # 1. Get Image Features
        # image should be (B, C, H, W) preprocessed
        # BLIP visual encoder:
        image_features = self.blip_model.visual_encoder(image) 
        # image_features shape: (B, num_patches+1, 768) - it's ViT output
        # We usually take the first token [CLS] for global representation
        image_features = image_features[:, 0, :] 
        
        # Normalize
        image_features = F.normalize(image_features, dim=-1)
        
        # 2. Get Text Features (Prompts)
        prompts_embeds, prompts_mask = self.prompt_learner() # (n_cls, len, dim)
        
        # Pass through Text Encoder
        # XBertEncoder forward supports inputs_embeds ?
        # Standard BertModel: forward(inputs_embeds=..., attention_mask=...)
        # We need to verify if 'text_encoder' is strictly BertModel.
        # It is XBertEncoder -> inherits BertModel.
        
        text_output = self.blip_model.text_encoder(
            inputs_embeds=prompts_embeds,
            attention_mask=prompts_mask,
            return_dict=True,
            mode='text' # specific to XBertEncoder to avoid cross-modal layers if existing?
            # Actually XBertEncoder standard forward is text-only if encoder_hidden_states is None
        )
        # Output: last_hidden_state (B, Len, Dim) and pooler_output (B, Dim)
        # We use pooler_output or [CLS] token? 
        # CLIP uses [EOT]. BERT uses [CLS] (index 0) or pooler (tanh(dense(CLS))).
        # BLIP typically uses [CLS] with projection. 
        # Feature extractor uses: 
        # text_output.last_hidden_state[:, 0, :] + projection
        
        text_features = text_output.last_hidden_state[:, 0, :]
        
        # Align Output Dimension?
        # Image feature dim: 768 (ViT-B) -> proj -> 256 (BLIP default proj)
        # Text feature dim: 768 (BERT-B) -> proj -> 256
        
        # We need the Projection Layers!
        # blip_model.vision_proj (Linear)
        # blip_model.text_proj (Linear)
        
        image_features = self.blip_model.vision_proj(image_features)
        text_features = self.blip_model.text_proj(text_features)
        
        # Normalize again after projection
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # 3. Compute Logits
        # Cosine similarity
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        
        return logits


@TRAINER_REGISTRY.register()
class APT_BLIP(TrainerX):
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading BLIP")
        # Ensure we are on correct device
        # load_model_and_preprocess puts it on device specified.
        # We might want to load cpu first then move to self.device
        blip_model, _ = load_blip_to_cpu(self.device)
        
        # We discard the processors because Dataloader already preprocesses?
        # WAIT. APT Dataloader uses CLIP transforms usually. 
        # BLIP transforms are different (e.g. 384x384).
        # We must override the dataloader transforms or ensure config matches.
        # For now, we assume user sets CFG input size correctly or we force it.
        # BLIP Base often uses 224, but Large uses 384. 
        # Let's assume standard 224 for now if not specified.
        
        self.model = CustomBLIP(cfg, classnames, blip_model, self.device)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name and "logit_scale" not in name:
                param.requires_grad_(False)
        
        # Ensure model is on device
        self.model.to(self.device)
        
        # Optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
        
        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        
        output = self.model(image)
        loss = F.cross_entropy(output, label)
        
        self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

