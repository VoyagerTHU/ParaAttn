import torch
import torch.distributed as dist
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.utils import export_to_video

import sys
import os

# 添加项目根目录到Python路径，确保能找到jintao模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 验证是否使用本地 para_attn
# /mnt/vepfs/base2/chenyixiao/yingze/ParaAttention/para_attn 绝对路径导入
# sys.path.append("/mnt/vepfs/base2/chenyixiao/yingze/ParaAttention/para_attn")
import para_attn
print(f"Using para_attn from: {para_attn.__file__}")

# from modify_hunyuan import set_sage_attn_hunyuan
import torch.nn.functional as F


from jintao.core import jintao_sage

dist.init_process_group()

torch.cuda.set_device(dist.get_rank())

# [rank1]: RuntimeError: Expected mha_graph->execute(handle, variant_pack, workspace_ptr.get()).is_good() to be true, but got false.  (Could this error message be improved?  If so, please report an enhancement request to PyTorch.)
# torch.backends.cuda.enable_cudnn_sdp(False)

# ATTENTION = {
#     "sage": jintao_sage,
#     "sdpa": F.scaled_dot_product_attention,
# }

model_id = "hunyuanvideo-community/HunyuanVideo"
# model_id = "tencent/HunyuanVideo"
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

# set_sage_attn_hunyuan(pipe.transformer, ATTENTION["sage"])

mesh = init_context_parallel_mesh(
    pipe.device.type,
)
print(f"mesh.shape: {mesh.shape}")
print(f"mesh.batch: {mesh['batch']}")
print(f"mesh.ring: {mesh['ring']}")
print(f"mesh.ulysses: {mesh['ulysses']}")

lis = [
    'An animated porcupine with a mix of brown and white fur and prominent quills is seen in a cozy, warmly lit interior setting, interacting with a green gift box with a yellow ribbon. The room is filled with wooden furniture and colorful wall decorations, suggesting a cheerful and domestic atmosphere. The porcupine\'s large eyes and expressive face convey a sense of lightheartedness and curiosity. The camera maintains a low angle, close to the ground, providing an intimate view of the character\'s actions without any movement, focusing on the playful and curious mood of the scene. The visual style is characteristic of contemporary 3D animation, with vibrant colors and smooth textures that create a polished and engaging look. The scene transitions to an outdoor environment, showcasing a sunny, verdant landscape with rocks, trees, and grass, indicating a natural, possibly forest-like setting. The presence of a small character in the final frame suggests the continuation of a narrative or the introduction of new characters.',
    'Animated characters, a rabbit and a mouse, are depicted in a perilous situation, first plummeting through a dark, undefined space, and then floating and swimming in a serene underwater environment. The characters are dressed in adventure gear, suggesting a narrative context. The camera closely follows their expressions and movements, capturing the tension and urgency of their situation. The medium and close-up shots emphasize their facial expressions, which convey fear and determination. The visual style is high-quality 3D animation with detailed textures and lighting, creating a cinematic feel.',
    'A man with facial hair, dressed in a burgundy shirt, is seen knocking on a weathered wooden door with a metal latch and a small window, set in a stone wall. The scene transitions to an indoor setting where the man, now wearing a blue shirt, speaks to the camera in a well-lit room furnished with a couch, a bookshelf, and various decorations. The video captures the man in a medium shot with a stationary camera, conveying a casual and friendly atmosphere in the indoor scene, contrasted with a neutral atmosphere in the outdoor scene. The visual style is realistic with natural lighting and color grading.',
    'Animated characters are engaging in a magical interaction within a dark, cavernous environment. The scene centers on a small, orange magical creature with a glowing heart, as well as two dragon-like creatures, one of which is holding a magical potion. The creature opens the potion, causing a transformation, which captures the attention of the dragons. Subsequently, two human characters with a torch discover the aftermath of the transformation, revealing a small, glowing creature resembling the one from earlier. The atmosphere is whimsical and magical, with a sense of curiosity and discovery. The camera remains static, offering medium shots that focus on the characters and their actions, while the visual style is traditional animation with smooth lines and vibrant colors.',
    'An animated character with white hair and a muscular build is shown in a close-up, displaying a stern and intense expression. The character is dressed in a red and gold outfit, suggesting a regal or powerful status. The scene transitions to reveal the character seated on a throne-like structure with ornate decorations, addressing a group of people who are standing in front of it. The atmosphere is serious and charged with emotion, indicating a moment of significance or decision-making. The camera focuses on the character\'s face before widening the shot to include the character\'s interaction with the group, using fixed position shots without any discernible camera movement. The visual style is characteristic of Japanese anime, with detailed character designs and vibrant coloring.'
]

parallelize_pipe(
    pipe,
    mesh=mesh,
    new_attention=jintao_sage,
    attention_args={
        'xpos_xi': 0.9999934149894527,
        'sigmoid_a': 1.0,
    },
    block_id=0,
    time_step=0,
    attention_type='repeat',
    method='thres',
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

for prompt in lis:
    output = pipe(
        prompt=prompt,
        height=544,
        width=960,
        num_frames=393,
        num_inference_steps=30,
        output_type="pil" if dist.get_rank() == 0 else "pt",
        generator = torch.Generator(device="cuda").manual_seed(42),
    ).frames[0]

    if dist.get_rank() == 0:
        print("Saving video to hunyuan_video_text_false_length.mp4")
        export_to_video(output, f"hunyuan_video_xpos_{'_'.join(prompt.split(' ')[0:10])}.mp4", fps=24)

dist.destroy_process_group()
