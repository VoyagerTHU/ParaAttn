import torch
import torch.distributed as dist
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.utils import export_to_video

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modify_hunyuan import set_sage_attn_hunyuan
import torch.nn.functional as F

dist.init_process_group()

torch.cuda.set_device(dist.get_rank())

# [rank1]: RuntimeError: Expected mha_graph->execute(handle, variant_pack, workspace_ptr.get()).is_good() to be true, but got false.  (Could this error message be improved?  If so, please report an enhancement request to PyTorch.)
# torch.backends.cuda.enable_cudnn_sdp(False)

from jintao.core import jintao_sage

ATTENTION = {
    "sage": jintao_sage,
    "sdpa": F.scaled_dot_product_attention,
}

model_id = "hunyuanvideo-community/HunyuanVideo"
transformer = HunyuanVideoTransformer3DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
    # revision="refs/pr/18",
)
pipe = HunyuanVideoPipeline.from_pretrained(
    model_id,
    transformer=transformer,
    torch_dtype=torch.float16,
    # revision="refs/pr/18",
).to("cuda")

from para_attn.context_parallel import init_context_parallel_mesh
from para_attn.context_parallel.diffusers_adapters import parallelize_pipe
from para_attn.parallel_vae.diffusers_adapters import parallelize_vae

set_sage_attn_hunyuan(pipe.transformer, ATTENTION["sage"])

mesh = init_context_parallel_mesh(
    pipe.device.type,
)
parallelize_pipe(
    pipe,
    mesh=mesh,
)
parallelize_vae(pipe.vae, mesh=mesh._flatten())

# from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe

# apply_cache_on_pipe(pipe)

# Enable memory savings
# pipe.enable_model_cpu_offload(gpu_id=dist.get_rank())
pipe.vae.enable_tiling(
    # Make it runnable on GPUs with 48GB memory
    # tile_sample_min_height=128,
    # tile_sample_stride_height=96,
    # tile_sample_min_width=128,
    # tile_sample_stride_width=96,
    # tile_sample_min_num_frames=32,
    # tile_sample_stride_num_frames=24,
)

# torch._inductor.config.reorder_for_compute_comm_overlap = True
# pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune-no-cudagraphs")

output = pipe(
    prompt="A cat walks on the grass, realistic",
    height=720,
    width=1280,
    num_frames=129,
    num_inference_steps=30,
    output_type="pil" if dist.get_rank() == 0 else "pt",
).frames[0]

if dist.get_rank() == 0:
    print("Saving video to hunyuan_video.mp4")
    export_to_video(output, "hunyuan_video.mp4", fps=15)

dist.destroy_process_group()
