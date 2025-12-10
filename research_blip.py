
import sys
import torch
import types

# Monkey patch transformers to support lavis
import transformers.modeling_utils as modeling_utils

if not hasattr(modeling_utils, "apply_chunking_to_forward"):
    print("Patching apply_chunking_to_forward in transformers...")
    def apply_chunking_to_forward(forward_fn, chunk_size, chunk_dim, *args):
        """
        This function applies the chunking to the forward function.
        """
        if chunk_size > 0:
            assert (
                args[0].shape[chunk_dim] % chunk_size == 0
            ), f"The dimension to be chunked {chunk_dim} has to be a multiple of the chunk size {chunk_size}"
            
            num_chunks = args[0].shape[chunk_dim] // chunk_size
            chunked_args = [torch.chunk(arg, num_chunks, dim=chunk_dim) for arg in args]
            
            outputs = [forward_fn(*chunk_args) for chunk_args in zip(*chunked_args)]
            # Stack results
            if isinstance(outputs[0], tuple):
                return tuple(
                    torch.cat([output[i] for output in outputs], dim=chunk_dim)
                    for i in range(len(outputs[0]))
                )
            return torch.cat(outputs, dim=chunk_dim)
            
        return forward_fn(*args)

    modeling_utils.apply_chunking_to_forward = apply_chunking_to_forward

if not hasattr(modeling_utils, "find_pruneable_heads_and_indices"):
    print("Patching find_pruneable_heads_and_indices in transformers...")
    def find_pruneable_heads_and_indices(heads, n_heads, head_mask, match_layer_norm_to_last_attention):
         # Minimal dummy implementation or copy from older transformers
         # Since we likely aren't pruning, returning empty lists might suffice, 
         # but let's see if we can perform a simple logic
         if head_mask is None:
             return set(), torch.arange(n_heads)
         # If simpler logic is needed, we can just return what we have
         # This is a complex function to polyfill correctly without full context.
         # For now let's hope it's not CALLED, just IMPORTED.
         # Actually it is imported.
         return set(), torch.arange(n_heads)
         
    modeling_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices

if not hasattr(modeling_utils, "prune_linear_layer"):
    print("Patching prune_linear_layer in transformers...")
    def prune_linear_layer(layer, index, dim=0):
        # Dummy implementation
        print("Warning: prune_linear_layer called (patched dummy)")
        return layer
        
    modeling_utils.prune_linear_layer = prune_linear_layer

try:
    from lavis.models import load_model_and_preprocess
    print("Successfully imported lavis!")
except Exception as e:
    print(f"Failed to import lavis: {e}")
    sys.exit(1)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading BLIP on {device}...")

try:
    # Load BLIP feature extractor
    model, vis_processors, txt_processors = load_model_and_preprocess(
        name="blip_feature_extractor", model_type="base", is_eval=True, device=device
    )

    print("\nModel Architecture:")
    # print(model)

    print("\nChecking Text Encoder keys...")
    if hasattr(model, "text_encoder"):
        print("Text Encoder found via .text_encoder")
        te = model.text_encoder
        print(te)
        # Check bert embedding
        if hasattr(te, "embeddings"):
            print("\nEmbeddings layer found.")
            print(te.embeddings.word_embeddings)
    else:
        print("Direct access to text_encoder failed.")

except Exception as e:
    print(f"Error loading model: {e}")
