import torch
import torch.nn.functional as F
from typing import Callable, List, Optional, Tuple, Union
from diffusers.models.attention_processor import Attention
from diffusers.models import HunyuanVideoTransformer3DModel
from functools import partial


class HunyuanVideoAttnProcessor2_0:
    def __init__(self, attn_func):
        self.attn_func = attn_func
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("HunyuanVideoAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply rotary embeddings if provided
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        # Use the provided attention function (either SageAttention or SDPA)
        hidden_states = self.attn_func(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        # Linear projection and dropout
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


def set_sage_attn_hunyuan(
        model: HunyuanVideoTransformer3DModel,
        attn_func,
):
    """
    Apply SageAttention to HunyuanVideo transformer blocks.
    
    Args:
        model: HunyuanVideoTransformer3DModel instance
        attn_func: The attention function to use (sageattn or F.scaled_dot_product_attention)
    """
    
    def recursive_apply_attention(module, name=""):
        """Recursively apply attention processor to all attention modules"""
        for child_name, child_module in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name
            
            # Check if this is an attention module
            if isinstance(child_module, Attention):
                processor = HunyuanVideoAttnProcessor2_0(attn_func)
                child_module.processor = processor
                print(f"Applied SageAttention to: {full_name}")
            else:
                # Recursively apply to child modules
                recursive_apply_attention(child_module, full_name)
    
    # Apply to the entire model recursively
    recursive_apply_attention(model)
    
    print(f"Applied SageAttention to all attention layers in HunyuanVideo transformer") 