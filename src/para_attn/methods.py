import torch

def pad_qkv(input_tensor, block_size=128):
    """
    Pad the input tensor to be a multiple of the block size.
    input shape: (seqlen, num_heads, hidden_dim)
    """
    bsz, num_heads,seqlen, hidden_dim = input_tensor.shape
    # Calculate the necessary padding
    padding_length = (block_size - (seqlen % block_size)) % block_size
    # Create a padded tensor with zeros
    padded_tensor = torch.zeros((bsz,num_heads,seqlen + padding_length,  hidden_dim), device=input_tensor.device, dtype=input_tensor.dtype)
    # Copy the original tensor into the padded tensor
    padded_tensor[:,:,:seqlen, :] = input_tensor
    
    return padded_tensor


def get_block_bias(
    block_avg,
    block_bias_args
):
    b, h, block_num, _ = block_avg.shape
    block_bias = torch.ones((b,h,block_num,block_num),dtype=torch.float16,device=block_avg.device)
    
    ref_function = block_bias_args['ref_function'] # of shape (block_num,block_num)
    # broadcast ref_function to the shape of block_avg
    ref_function = ref_function.unsqueeze(0).unsqueeze(0).expand(b,h,block_num,block_num).to(block_avg.device)
    
    eps = 1e-8
    block_avg_proportion = block_avg / 128.0
    block_bias = torch.div(ref_function, block_avg_proportion + eps).clamp(min=0.0, max=1.0)
    block_bias = torch.log2(block_bias)

    # 如果有-inf，替代成-1e4的大负数
    block_bias = torch.where(torch.isinf(block_bias), torch.full_like(block_bias, -1e4), block_bias)
    # print(block_bias[:,:,300,:])
    
    return block_bias

   
def get_decay_mask(
    block_avg,
    decay_mask_threshold
):
    b,h,block_num,_ = block_avg.shape
    block_avg_proportion = block_avg / 128.0
    decay_mask = torch.ones((b,h,block_num,block_num),dtype=torch.bool,device=block_avg.device)
    decay_mask = decay_mask * (block_avg_proportion < decay_mask_threshold)
    # print(f"decay_mask[0,0,300,:100]: {decay_mask[0,0,300,:100]}")
    # print(f"decay_mask[0,0,300,-100:]: {decay_mask[0,0,300,-100:]}")
    # print(f"decay_mask.sum(dim=-1)[0,0,:100]: {decay_mask.sum(dim=-1)[0,0,:100]}")
    return decay_mask    