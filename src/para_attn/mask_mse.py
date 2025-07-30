import torch
import math
import torch.nn.functional as F
from .jintao.constants import *
def get_attention_mask_legacy(mask_name, sample_mse_max_row, context_length, num_frame, frame_size, width, is_Hunyuan=False):
    
    # from termcolor import colored

    # allocated = torch.cuda.memory_allocated() / 1e9
    # print(colored(f"Allocated Memory: {allocated:.2f} GB", "yellow"))

    attention_mask = torch.zeros((context_length + num_frame * frame_size, context_length + num_frame * frame_size), device="cpu")

    # TODO: fix hard coded mask
    if mask_name == "spatial":
        pixel_attn_mask = torch.zeros_like(attention_mask, dtype=torch.bool, device="cpu")
        
        pixel_attn_mask[:, :frame_size] = 1 # First Frame Sink
        
        block_size, block_thres = 128, frame_size * width
        num_block = math.ceil(num_frame * frame_size / block_size)
        for i in range(num_block):
            for j in range(num_block):
                if abs(i - j) < block_thres // block_size:
                    pixel_attn_mask[i * block_size : (i + 1) * block_size, j * block_size : (j + 1) * block_size] = 1
        attention_mask = pixel_attn_mask
    else:
        pixel_attn_mask = torch.zeros_like(attention_mask, dtype=torch.bool, device="cpu")

        pixel_attn_mask[:, :frame_size] = 1 # First Frame Sink
        
        block_size, block_thres = 128, frame_size * width
        num_block = math.ceil(num_frame * frame_size / block_size)
        for i in range(num_block):
            for j in range(num_block):
                if abs(i - j) < block_thres // block_size:
                    pixel_attn_mask[i * block_size : (i + 1) * block_size, j * block_size : (j + 1) * block_size] = 1

        pixel_attn_mask = pixel_attn_mask.reshape(frame_size, num_frame, frame_size, num_frame).permute(1, 0, 3, 2).reshape(frame_size * num_frame, frame_size * num_frame)
        attention_mask = pixel_attn_mask

    attention_mask = attention_mask[:sample_mse_max_row].cuda()
    return attention_mask

def get_attention_mask_legacy_2(mask_name, sample_mse_max_row, context_length, num_frame, frame_size, width, is_Hunyuan=False, num_selected_rows=512, return_indices=False):
    """
    Generate attention mask by selecting num_selected_rows from all rows instead of taking the first sample_mse_max_row rows.
    
    Args:
        mask_name: Type of mask ("spatial" or other)
        sample_mse_max_row: Maximum number of rows (for compatibility, but not used for row selection)
        context_length: Length of context
        num_frame: Number of frames
        frame_size: Size of each frame
        width: Width parameter for block threshold
        is_Hunyuan: Whether using Hunyuan model
        num_selected_rows: Number of rows to select from all rows (default: 512)
        return_indices: Whether to return the selected indices along with the mask
    
    Returns:
        attention_mask: Selected attention mask on CUDA
        selected_indices: (optional) The indices of selected rows if return_indices=True
    """
    
    # Create full attention mask
    print(f"context_length: {context_length}")
    full_size = context_length + num_frame * frame_size
    print(f"full_size: {full_size}")
    attention_mask = torch.zeros((full_size, full_size), device="cuda")
    print(f"attention_mask: {attention_mask.shape}")

    # Generate mask pattern similar to legacy version
    if mask_name == "spatial":
        print(f"mask_name: {mask_name}")
        pixel_attn_mask = torch.zeros_like(attention_mask, dtype=torch.bool, device="cpu")
        
        pixel_attn_mask[:, :frame_size] = 1 # First Frame Sink
        print(f"pixel_attn_mask: {pixel_attn_mask.shape}")
        block_size, block_thres = 128, frame_size * width
        num_block = math.ceil(num_frame * frame_size / block_size)
        print(f"num_block: {num_block}")
        for i in range(num_block):
            for j in range(num_block):
                if abs(i - j) < block_thres // block_size:
                    pixel_attn_mask[i * block_size : (i + 1) * block_size, j * block_size : (j + 1) * block_size] = 1
                    print(f"pixel_attn_mask in pos: {i}, {j}")
        attention_mask = pixel_attn_mask
        print(f"attention_mask: {attention_mask.shape}")
    else:
        print(f"mask_name not spatial: {mask_name}")
        pixel_attn_mask = torch.zeros_like(attention_mask, dtype=torch.bool, device="cpu")

        pixel_attn_mask[:, :frame_size] = 1 # First Frame Sink
        
        block_size, block_thres = 128, frame_size * width
        num_block = math.ceil(num_frame * frame_size / block_size)
        for i in range(num_block):
            for j in range(num_block):
                if abs(i - j) < block_thres // block_size:
                    pixel_attn_mask[i * block_size : (i + 1) * block_size, j * block_size : (j + 1) * block_size] = 1

        pixel_attn_mask = pixel_attn_mask.reshape(frame_size, num_frame, frame_size, num_frame).permute(1, 0, 3, 2).reshape(frame_size * num_frame, frame_size * num_frame)
        attention_mask = pixel_attn_mask

    # Select num_selected_rows from all rows instead of taking the first sample_mse_max_row rows
    total_rows = attention_mask.shape[0]
    num_selected_rows = min(num_selected_rows, total_rows)
    print(f"num_selected_rows: {num_selected_rows}")
    # Generate indices for row selection (can be random, uniform, or other strategies)
    # Here using uniform sampling across all rows
    selected_indices = torch.linspace(0, total_rows - 1, num_selected_rows).long()
    print(f"selected_indices: {selected_indices}")
    # Alternative: random sampling
    # selected_indices = torch.randperm(total_rows)[:num_selected_rows].sort()[0]
    
    # Select the specified rows
    selected_attention_mask = attention_mask[selected_indices].cuda()
    
    if return_indices:
        return selected_attention_mask, selected_indices
    else:
        return selected_attention_mask
    
def get_attention_mask(mask_name, sample_mse_max_row, context_length, num_frame, frame_size, width, is_Hunyuan=False, num_selected_rows=512, return_indices=False):
    """
    Generate attention mask by directly computing only num_selected_rows instead of full mask.
    Uses linspace sampling to select rows.
    
    Args:
        mask_name: Type of mask ("spatial" or "temporal")
        sample_mse_max_row: Maximum number of rows (for compatibility)
        context_length: Length of context
        num_frame: Number of frames
        frame_size: Size of each frame
        width: Width parameter for block threshold
        is_Hunyuan: Whether using Hunyuan model
        num_selected_rows: Number of rows to compute (default: 512)
        return_indices: Whether to return the selected indices along with the mask
    
    Returns:
        attention_mask: Selected attention mask on CUDA
        selected_indices: (optional) The indices of selected rows if return_indices=True
    """
    
    full_size = context_length + num_frame * frame_size
    # print(f"full_size: {full_size}")
    num_selected_rows = min(num_selected_rows, full_size)
    
    # Generate selected row indices using linspace
    if not is_Hunyuan:
        selected_indices = torch.linspace(context_length, full_size - 1, num_selected_rows).long()
    else:
        selected_indices = torch.linspace(0, full_size - context_length - 1, num_selected_rows).long()

    # Initialize mask for selected rows only
    attention_mask = torch.zeros((num_selected_rows, full_size), dtype=torch.bool, device="cpu")
    
    if mask_name == "spatial":
        # First Frame Sink for selected rows
        attention_mask[:, :frame_size] = 1
        
        # Compute spatial blocks only for selected rows
        block_size, block_thres = 128, frame_size * width
        num_block = math.ceil(num_frame * frame_size / block_size)
        
        if not is_Hunyuan:
            for idx, row_idx in enumerate(selected_indices):
                # Skip context rows (they already have first frame sink)
                if row_idx < context_length:
                    continue
                    
                # Calculate which block this row belongs to
                pixel_row = row_idx - context_length
                row_block = pixel_row // block_size
                
                # Set attention for nearby blocks
                for j in range(num_block):
                    if abs(row_block - j) < block_thres // block_size:
                        start_col = context_length + j * block_size
                        end_col = min(context_length + (j + 1) * block_size, full_size)
                        attention_mask[idx, start_col:end_col] = 1
        else:
            for idx, row_idx in enumerate(selected_indices):
                pixel_row = row_idx
                row_block = pixel_row // block_size
                
                # Set attention for nearby blocks
                for j in range(num_block):
                    if abs(row_block - j) < block_thres // block_size:
                        start_col =  j * block_size
                        end_col = min((j + 1) * block_size, full_size)
                        attention_mask[idx, start_col:end_col] = 1
                        
    
    elif mask_name == "temporal":
        # # First Frame Sink for selected rows
        # attention_mask[:, :frame_size] = 1
        
        # # Compute temporal blocks only for selected rows
        # block_size, block_thres = 128, frame_size * width
        # num_block = math.ceil(num_frame * frame_size / block_size)
        
        # for idx, row_idx in enumerate(selected_indices):
        #     # Skip context rows
        #     if row_idx < context_length:
        #         continue
                
        #     pixel_row = row_idx - context_length
        #     row_block = pixel_row // block_size
            
        #     # Set attention for nearby blocks
        #     for j in range(num_block):
        #         if abs(row_block - j) < block_thres // block_size:
        #             start_col = context_length + j * block_size
        #             end_col = min(context_length + (j + 1) * block_size, full_size)
        #             attention_mask[idx, start_col:end_col] = 1
        
        # # Apply temporal permutation pattern for selected rows
        # temp_mask = torch.zeros_like(attention_mask)
        # for idx, row_idx in enumerate(selected_indices):
        #     if row_idx < context_length:
        #         temp_mask[idx] = attention_mask[idx]
        #         continue
                
        #     pixel_row = row_idx - context_length
        #     frame_idx = pixel_row // frame_size
        #     pixel_in_frame = pixel_row % frame_size
            
        #     # Apply temporal permutation
        #     for col_idx in range(context_length, full_size):
        #         col_pixel = col_idx - context_length
        #         col_frame = col_pixel // frame_size
        #         col_pixel_in_frame = col_pixel % frame_size
                
        #         # Map to permuted position
        #         permuted_col = context_length + col_frame * frame_size + pixel_in_frame
        #         if permuted_col < full_size and attention_mask[idx, permuted_col]:
        #             temp_mask[idx, col_idx] = 1
        
        # attention_mask = temp_mask
        attention_mask = torch.zeros_like(attention_mask)
    
    # Move to CUDA
    attention_mask = attention_mask.cuda()
    
    if return_indices:
        return attention_mask, selected_indices
    else:
        return attention_mask

def sample_mse(query, key, value, attention_masks, num_sampled_rows):
    assert len(attention_masks) == 2

    cfg, num_heads, seq_len, dim = query.size()
    num_sampled_rows = min(num_sampled_rows, seq_len)
    # sample_mse_max_row = 
    # print("shape for attn mask", [mask.shape for mask in attention_masks])
    # print("shape for query", query.shape)
    sample_mse_max_row = attention_masks[0].shape[0]
    # print("sample_mse_max_row", sample_mse_max_row)
    sampled_rows = torch.randint(low=0, high=sample_mse_max_row, size=(num_sampled_rows,))
    sampled_q = query[:, :, sampled_rows, :]
    sampled_qk_scores = torch.matmul(sampled_q, key.transpose(-2, -1)) / (dim**0.5)

    sampled_attn_weights = F.softmax(sampled_qk_scores, dim=-1)
    sampled_golden_hidden_states = torch.matmul(sampled_attn_weights, value)  # (1, seq_len, dim)

    sampled_mses = torch.zeros(len(attention_masks), cfg, num_heads, device=query.device, dtype=query.dtype)

    # Only have Tri-diagonal and Striped
    for mask_idx, attn_mask in enumerate(attention_masks):
        sampled_attention_mask = attn_mask[sampled_rows, :]
        sampled_attention_scores = sampled_qk_scores.masked_fill(sampled_attention_mask == 0, float('-inf'))
        sampled_attn_weights = F.softmax(sampled_attention_scores, dim=-1)
        sampled_hidden_states = torch.matmul(sampled_attn_weights, value)
        mse = torch.mean((sampled_hidden_states - sampled_golden_hidden_states) ** 2, dim=(2, 3))
        sampled_mses[mask_idx] = mse

    return sampled_mses

def direct_threshold(query, key, value, attention_masks_narrow, attention_masks_wide, num_sampled_rows, mask_selected_indices=None, threshold=0.5):
    cfg, num_heads, seq_len, dim = query.size()
    num_sampled_rows = min(num_sampled_rows, seq_len)
    
    # Get the number of rows available in the attention masks
    sample_mse_max_row = attention_masks_narrow[0].shape[0]
    
    # If mask_selected_indices is provided, it means we're using the new get_attention_mask
    # and need to sample from the pre-selected rows
    if mask_selected_indices is not None:
        # Sample from the pre-selected indices
        if num_sampled_rows <= len(mask_selected_indices):
            # Uniformly sample from the pre-selected indices
            sampled_indices_in_mask = torch.linspace(0, sample_mse_max_row - 1, num_sampled_rows).long()
            # Map back to original sequence indices
            sampled_rows = mask_selected_indices[sampled_indices_in_mask]
        else:
            # If we need more samples than available in mask, use all mask rows
            sampled_rows = mask_selected_indices
            sampled_indices_in_mask = torch.arange(sample_mse_max_row)
    else:
        # Legacy behavior: sample uniformly from consecutive rows
        sampled_rows = torch.linspace(0, sample_mse_max_row - 1, num_sampled_rows).long()
        sampled_indices_in_mask = sampled_rows
    
    sampled_q = query[:, :, sampled_rows, :]
    sampled_qk_scores = torch.matmul(sampled_q, key.transpose(-2, -1)) / (dim**0.5)
    
    sampled_attn_weights = F.softmax(sampled_qk_scores, dim=-1)
    # print(f"sampled_attn_weights: {sampled_attn_weights.shape}")
    
    # 初始化flags为TEMPORAL（其他情况）
    flags = torch.full((cfg, num_heads), TEMPORAL, dtype=torch.long, device=query.device)
    
    # 检查narrow mask - use the indices within the mask
    attn_mask_narrow = attention_masks_narrow[0]
    # print(f"attn_mask_narrow: {attn_mask_narrow.shape}")
    # sampled_attention_mask_narrow = attn_mask_narrow[sampled_indices_in_mask, :]
    sampled_attention_mask_narrow = attn_mask_narrow
    masked_qk_weights_narrow = sampled_attn_weights.masked_fill(sampled_attention_mask_narrow == 0, 0)
    masked_qk_weights_sum_narrow = torch.sum(masked_qk_weights_narrow, dim=-1).mean(dim=-1)  # (cfg, num_heads)
    
    # 如果narrow时>0.7，设置为0
    narrow_condition = masked_qk_weights_sum_narrow > threshold
    flags[narrow_condition] = SPATIAL
    
    # 检查wide mask（只对不满足narrow条件的进行检查）- use the indices within the mask
    attn_mask_wide = attention_masks_wide[0]
    sampled_attention_mask_wide = attn_mask_wide[sampled_indices_in_mask, :]
    masked_qk_weights_wide = sampled_attn_weights.masked_fill(sampled_attention_mask_wide == 0, 0)
    masked_qk_weights_sum_wide = torch.sum(masked_qk_weights_wide, dim=-1).mean(dim=-1)  # (cfg, num_heads)
    
    # 如果wide时>0.7且不满足narrow条件，设置为1
    wide_condition = (masked_qk_weights_sum_wide > threshold) & (~narrow_condition)
    flags[wide_condition] = MID

    
    # 其余情况保持为2
    return flags


def direct_threshold_proportional(query, key, attention_masks_narrow, num_sampled_rows, mask_selected_indices=None, threshold=0.5):
    cfg, num_heads, seq_len, dim = query.size()
    num_sampled_rows = min(num_sampled_rows, seq_len)

    # Get the number of rows available in the attention masks
    sample_mse_max_row = attention_masks_narrow[0].shape[0]
    
    # If mask_selected_indices is provided, it means we're using the new get_attention_mask
    # and need to sample from the pre-selected rows
    if mask_selected_indices is not None:
        if num_sampled_rows <= len(mask_selected_indices):
            sampled_indices_in_mask = torch.linspace(0, sample_mse_max_row - 1, num_sampled_rows).long()
            sampled_rows = mask_selected_indices[sampled_indices_in_mask]
        else:
            sampled_rows = mask_selected_indices
            sampled_indices_in_mask = torch.arange(sample_mse_max_row)
    else:
        sampled_rows = torch.linspace(0, sample_mse_max_row - 1, num_sampled_rows).long()
        sampled_indices_in_mask = sampled_rows
    
    sampled_q = query[:, :, sampled_rows, :]
    sampled_qk_scores = torch.matmul(sampled_q, key.transpose(-2, -1)) / (dim**0.5)
    
    sampled_attn_weights = F.softmax(sampled_qk_scores, dim=-1)

    flags = torch.zeros((cfg, num_heads), dtype=torch.long, device=query.device)
    
    # 检查narrow mask - use the indices within the mask

    sampled_attention_mask_narrow = attention_masks_narrow[0]
    masked_qk_weights_narrow = sampled_attn_weights.masked_fill(sampled_attention_mask_narrow == 0, 0)
    masked_qk_weights_sum_narrow = torch.sum(masked_qk_weights_narrow, dim=-1).mean(dim=-1)  # (cfg, num_heads)

    narrow_condition = masked_qk_weights_sum_narrow > threshold
    flags[narrow_condition] = 1

    return flags
    
def get_spatial_temporal_flag(q, k, v, attention_masks_narrow, attention_masks_wide, num_sampled_rows, method="thres", mask_selected_indices=None, threshold=0.6):
    # sampled_mses = sample_mse(q, k, v, attention_masks, num_sampled_rows)
    # best_mask_idx = torch.argmin(sampled_mses, dim=0)
    # return best_mask_idx
    if method == "thres":
        flags = direct_threshold(q, k, v, attention_masks_narrow, attention_masks_wide, num_sampled_rows, mask_selected_indices, threshold=threshold)
    elif method == "mse":
        mses = sample_mse(q, k, v, attention_masks_narrow, num_sampled_rows)
        flags = torch.argmin(mses, dim=0)
    elif method == "simple":
        cfg, num_heads, seq_len, dim = q.size()
        flags = torch.zeros((cfg, num_heads), dtype=torch.long, device=q.device)
    elif method == "proportional":
        flags = direct_threshold_proportional(q, k, attention_masks_narrow, num_sampled_rows, mask_selected_indices, threshold=threshold)
    return flags

