import torch, math
import triton
import triton.language as tl
from .constants import *

# 预计算常数
# ALPHA_XPOS_XI = ((0.97)**(1/16000))
# BETA_XPOS_XI = ((0.8)**(1/16000))
# FRAME_TOKENS = 1560

@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q, q_scale, kv_len, current_flag,
                    K_ptrs, K_scale_ptr, V_ptrs, stride_kn, stride_vn, 
                    start_m,  
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,                                 
                    xpos_xi: tl.constexpr = 0.9999934149894527,
                    frame_tokens: tl.constexpr = 1560,
                    sigmoid_a: tl.constexpr = 1.0,
                    alpha_xpos_xi: tl.constexpr = 0.9999967941742395,
                    beta_xpos_xi: tl.constexpr = 0.9999860536252945
                    ):
    DEFAULT=0
    XPOS=1
    SIGMOID=2
    INTERPOLATION=3
    SPATIAL=4
    MID=5
    TEMPORAL=6
    
    LOG2_XPOS = tl.log2(tl.full((), xpos_xi, dtype=tl.float32))
    lo, hi = 0, kv_len
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k_mask = offs_n[None, :] < (kv_len - start_n)   
        k = tl.load(K_ptrs, mask = k_mask)
        k_scale = tl.load(K_scale_ptr)

        m = offs_m[:, None]
        n = start_n + offs_n
        
        qk = tl.dot(q, k).to(tl.float32) * q_scale * k_scale

        if  current_flag == DEFAULT:
            qk = qk
            
        if  current_flag == XPOS:
            dist    = tl.abs(m - n).to(tl.float32)             # |m-n|
            xpos_pow = tl.math.exp2(dist * LOG2_XPOS)          # xi^{|m-n|}
            qk *= xpos_pow
            
        # if current_flag==TEMPORAL:#current_flag == SPATIAL or current_flag == MID:
        #     dist    = tl.abs(m - n).to(tl.float32)             # |m-n|
        #     xpos_pow = tl.math.exp2(dist * LOG2_XPOS)          # xi^{|m-n|}
        #     qk *= xpos_pow
        
        if current_flag == SIGMOID:
            sigmoid_a_val = tl.full((), sigmoid_a, dtype=tl.float32)
            dist    = tl.abs(m - n).to(tl.float32)   
            # tl.device_assert(kv_len == 32000 * 4, "kv_len must be 32000 * 4")
            rela_dist = dist / (kv_len / 4)
            sigmoid_pow = (1 / (1 + tl.math.exp(sigmoid_a_val * (rela_dist - 1))) + 0.5) * 0.34 + 0.5
            qk *= sigmoid_pow

        if current_flag == INTERPOLATION:
            # 使用预计算的常数
            alpha_log = tl.log2(tl.full((), alpha_xpos_xi, dtype=tl.float32))
            beta_log = tl.log2(tl.full((), beta_xpos_xi, dtype=tl.float32))
            
            # 计算帧内位置 - 修复tl.full_like错误
            frame_tokens_scalar = tl.full((), frame_tokens, dtype=tl.float32)
            n_pos_within_frame = n % frame_tokens_scalar
            half_frame = frame_tokens_scalar / 2
            
            # 使用tl.where避免分支divergence
            mask = n_pos_within_frame < half_frame
            factor1 = 1 - n_pos_within_frame / half_frame
            factor2 = (n_pos_within_frame - half_frame) / half_frame
            interp_factor = tl.where(mask, factor1, factor2)
            
            # 计算距离和幂次
            dist = tl.abs(m - n).to(tl.float32)
            alpha_xpos_pow = tl.math.exp2(dist * alpha_log)
            beta_xpos_pow = tl.math.exp2(dist * beta_log)
            
            # 插值计算
            xpos_pow = interp_factor * alpha_xpos_pow + (1 - interp_factor) * beta_xpos_pow
            qk *= xpos_pow
        
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        
        acc = acc * alpha[:, None]
        
        v = tl.load(V_ptrs, mask = offs_n[:, None] < (kv_len - start_n))
        p = p.to(tl.float16)
        
        acc += tl.dot(p, v, out_dtype=tl.float16)   
        m_i = m_ij
        K_ptrs += BLOCK_N * stride_kn
        K_scale_ptr += 1
        V_ptrs += BLOCK_N * stride_vn
    return acc, l_i

@triton.jit
def _attn_fwd(Q, K, V, Q_scale, K_scale, Out,  
              flags, stride_f_b, stride_f_h,
              stride_qz, stride_qh, stride_qn,
              stride_kz, stride_kh, stride_kn,  
              stride_vz, stride_vh, stride_vn,  
              stride_oz, stride_oh, stride_on,  
              qo_len, kv_len, H: tl.constexpr, num_kv_groups: tl.constexpr,
              HEAD_DIM: tl.constexpr,  
              BLOCK_M: tl.constexpr,  
              BLOCK_N: tl.constexpr,  
              STAGE: tl.constexpr,     
              xpos_xi: tl.constexpr = 0.9999934149894527,
              frame_tokens: tl.constexpr = 1560,
              sigmoid_a: tl.constexpr = 1.0, 
              alpha_xpos_xi: tl.constexpr = 0.9999967941742395, 
              beta_xpos_xi: tl.constexpr = 0.9999860536252945
              ):
    start_m = tl.program_id(0)

    off_z = tl.program_id(2).to(tl.int64)
    off_h = tl.program_id(1).to(tl.int64)

    q_scale_offset = (off_z * H + off_h) * tl.cdiv(qo_len, BLOCK_M)
    k_scale_offset = (off_z * (H // num_kv_groups) + off_h // num_kv_groups) * tl.cdiv(kv_len, BLOCK_N) 

    flag_ptr = flags + off_z * stride_f_b + off_h * stride_f_h
    current_flag = tl.load(flag_ptr) 
    
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)
    Q_ptrs = Q + (off_z * stride_qz + off_h * stride_qh) + offs_m[:, None] * stride_qn + offs_k[None, :]
    Q_scale_ptr = Q_scale + q_scale_offset + start_m
    K_ptrs = K + (off_z * stride_kz + (off_h // num_kv_groups) * stride_kh) + offs_n[None, :] * stride_kn + offs_k[:, None] 
    K_scale_ptr = K_scale + k_scale_offset
    V_ptrs = V + (off_z * stride_vz + (off_h // num_kv_groups) * stride_vh) + offs_n[:, None] * stride_vn + offs_k[None, :]
    O_block_ptr = Out + (off_z * stride_oz + off_h * stride_oh) + offs_m[:, None] * stride_on + offs_k[None, :]
    
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    
    q = tl.load(Q_ptrs, mask = offs_m[:, None] < qo_len)
    q_scale = tl.load(Q_scale_ptr)
    acc, l_i = _attn_fwd_inner(acc, l_i, m_i, q, q_scale, kv_len, current_flag, K_ptrs, K_scale_ptr, V_ptrs, 
                               stride_kn, stride_vn,
                                    start_m,  
                                    BLOCK_M, HEAD_DIM, BLOCK_N,  
                                    4 - STAGE, offs_m, offs_n,
                                    xpos_xi=xpos_xi,
                                    frame_tokens=frame_tokens,
                                    sigmoid_a=sigmoid_a,
                                    alpha_xpos_xi=alpha_xpos_xi,
                                    beta_xpos_xi=beta_xpos_xi
                                    )
    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), mask = (offs_m[:, None] < qo_len))

def forward(q, k, v, flags, q_scale, k_scale, tensor_layout="HND", output_dtype=torch.float16,    
              xpos_xi: tl.constexpr = 0.9999934149894527, 
              frame_tokens: tl.constexpr = 1560,
              sigmoid_a: tl.constexpr = 1.0, 
              alpha_xpos_xi: tl.constexpr = 0.9999967941742395, 
              beta_xpos_xi: tl.constexpr = 0.9999860536252945):
    BLOCK_M = 128
    BLOCK_N = 64
    stage = 1

    o = torch.empty(q.shape, dtype=output_dtype, device=q.device)

    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape

        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(1), q.stride(2)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(1), k.stride(2)
        stride_bz_v, stride_h_v, stride_seq_v = v.stride(0), v.stride(1), v.stride(2)
        stride_bz_o, stride_h_o, stride_seq_o = o.stride(0), o.stride(1), o.stride(2)
    elif tensor_layout == "NHD":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape

        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(2), q.stride(1)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(2), k.stride(1)
        stride_bz_v, stride_h_v, stride_seq_v = v.stride(0), v.stride(2), v.stride(1)
        stride_bz_o, stride_h_o, stride_seq_o = o.stride(0), o.stride(2), o.stride(1)
    else:
        raise ValueError(f"tensor_layout {tensor_layout} not supported")
    
    stride_f_b, stride_f_h = flags.stride()
    
    HEAD_DIM_K = head_dim
    num_kv_groups = h_qo // h_kv

    grid = (triton.cdiv(qo_len, BLOCK_M), h_qo, b)
    _attn_fwd[grid](
        q, k, v, q_scale, k_scale, o,  
        flags,
        stride_f_b, stride_f_h,
        stride_bz_q, stride_h_q, stride_seq_q, 
        stride_bz_k, stride_h_k, stride_seq_k,  
        stride_bz_v, stride_h_v, stride_seq_v,  
        stride_bz_o, stride_h_o, stride_seq_o,
        qo_len, kv_len,
        h_qo, num_kv_groups,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=HEAD_DIM_K,  
        STAGE=stage,  
        num_warps=4 if head_dim == 64 else 8,
        num_stages=3 if head_dim == 64 else 4, 
        xpos_xi=xpos_xi, 
        frame_tokens=frame_tokens,
        sigmoid_a=sigmoid_a, 
        alpha_xpos_xi=alpha_xpos_xi, 
        beta_xpos_xi=beta_xpos_xi
        )
    return o