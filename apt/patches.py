import transformers.modeling_utils as modeling_utils
import torch

# ==============================================================================
# Patch transformers for salesforce-lavis compatibility
# ==============================================================================

if not hasattr(modeling_utils, "apply_chunking_to_forward"):
    print("[Patches] Patching apply_chunking_to_forward in transformers...")
    def apply_chunking_to_forward(forward_fn, chunk_size, chunk_dim, *args):
        if chunk_size > 0:
            assert (
                args[0].shape[chunk_dim] % chunk_size == 0
            ), f"The dimension to be chunked {chunk_dim} has to be a multiple of the chunk size {chunk_size}"
            
            num_chunks = args[0].shape[chunk_dim] // chunk_size
            chunked_args = [torch.chunk(arg, num_chunks, dim=chunk_dim) for arg in args]
            
            outputs = [forward_fn(*chunk_args) for chunk_args in zip(*chunked_args)]
            if isinstance(outputs[0], tuple):
                return tuple(
                    torch.cat([output[i] for output in outputs], dim=chunk_dim)
                    for i in range(len(outputs[0]))
                )
            return torch.cat(outputs, dim=chunk_dim)
        return forward_fn(*args)

    modeling_utils.apply_chunking_to_forward = apply_chunking_to_forward

if not hasattr(modeling_utils, "find_pruneable_heads_and_indices"):
    print("[Patches] Patching find_pruneable_heads_and_indices in transformers...")
    def find_pruneable_heads_and_indices(heads, n_heads, head_mask, match_layer_norm_to_last_attention):
         if head_mask is None:
             return set(), torch.arange(n_heads)
         return set(), torch.arange(n_heads)
         
    modeling_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices

if not hasattr(modeling_utils, "prune_linear_layer"):
    print("[Patches] Patching prune_linear_layer in transformers...")
    def prune_linear_layer(layer, index, dim=0):
        return layer
        
    modeling_utils.prune_linear_layer = prune_linear_layer

# ==============================================================================
# Patch dassl for PyTorch compatibility
# ==============================================================================
try:
    import dassl.optim.lr_scheduler as dls
    from torch.optim.lr_scheduler import _LRScheduler
    
    # Check if we need to patch (if init has 'verbose' but base class doesn't want it, 
    # or if we just want to be safe)
    # The error was: LRScheduler.__init__() takes ... but 4 were given (verbose passed)
    
    print("[Patches] Patching dassl.optim.lr_scheduler._BaseWarmupScheduler...")
    def new_init(self, optimizer, successor, warmup_epoch, last_epoch=-1, verbose=False):
        self.successor = successor
        self.warmup_epoch = warmup_epoch
        _LRScheduler.__init__(self, optimizer, last_epoch)
        
    dls._BaseWarmupScheduler.__init__ = new_init
except Exception as e:
    print(f"[Patches] Warning: Failed to patch dassl scheduler: {e}")

# ==============================================================================
# Patch dassl for PyTorch 2.6+ weights_only=True default
# ==============================================================================
try:
    import dassl.utils.torchtools as torchtools
    
    print("[Patches] Patching dassl.utils.torchtools.load_checkpoint (weights_only=False)...")
    _original_load_checkpoint = torchtools.load_checkpoint
    
    def new_load_checkpoint(fpath):
        # We assume all checkpoints loaded via this tool are safe
        try:
             return torch.load(fpath, map_location="cpu", weights_only=False)
        except TypeError:
             # Fallback for older torch versions if they don't have weights_only arg?
             # But we know we are on 2.6+ if we hit the previous error.
             return torch.load(fpath, map_location="cpu")

    torchtools.load_checkpoint = new_load_checkpoint
except Exception as e:
    print(f"[Patches] Warning: Failed to patch load_checkpoint: {e}")

