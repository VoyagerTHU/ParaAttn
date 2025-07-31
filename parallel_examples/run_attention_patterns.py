import torch
import torch.distributed as dist
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.utils import export_to_video
import sys
import os
import itertools
import json
from datetime import datetime
import torch.nn.functional as F

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import para_attn
from para_attn.jintao.core import jintao_sage
from para_attn.context_parallel import init_context_parallel_mesh
from para_attn.context_parallel.diffusers_adapters import parallelize_pipe
from para_attn.parallel_vae.diffusers_adapters import parallelize_vae
import gc

def init_distributed():
    """初始化分布式环境"""
    dist.init_process_group()
    torch.cuda.set_device(dist.get_rank())

def load_model():
    """加载模型"""
    model_id = "hunyuanvideo-community/HunyuanVideo"
    
    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )
    
    pipe = HunyuanVideoPipeline.from_pretrained(
        model_id,
        transformer=transformer,
        torch_dtype=torch.float16,
    ).to("cuda")
    
    return pipe

def get_attention_masks(attention_type, sample_mse_max_row, context_length, num_frames, frame_size, narrow_width, wide_width):
    """获取attention masks"""
    
    from para_attn.mask_mse import get_attention_mask
    
    masks = ['spatial', 'temporal']
    num_latent_frames = (num_frames - 1) // 4 + 1
    
    attention_masks_narrow = []
    attention_masks_wide = []
    
    for mask_name in masks:
        mask_narrow, mask_selected_indices = get_attention_mask(
            mask_name, sample_mse_max_row, context_length, num_latent_frames, 
            frame_size, width=narrow_width, num_selected_rows=sample_mse_max_row, 
            return_indices=True, is_Hunyuan=True
        )
        mask_wide, _ = get_attention_mask(
            mask_name, sample_mse_max_row, context_length, num_latent_frames, 
            frame_size, width=wide_width, num_selected_rows=sample_mse_max_row, 
            return_indices=True, is_Hunyuan=True
        )
        # print(f"mask_narrow: {mask_narrow.shape}")
        # print(f"mask_wide: {mask_wide.shape}")
        
        if mask_narrow is not None:
            attention_masks_narrow.append(mask_narrow.to("cuda"))
        if mask_wide is not None:
            attention_masks_wide.append(mask_wide.to("cuda"))
    
    return attention_masks_narrow, attention_masks_wide, mask_selected_indices


def get_ref_function(
    blocks_num: int,
    ref_name: str,
    k=0.0025,
    a=0.7,
    b=0.1,
    c=0.0001,
    r=10,
    window_size: int = 128
):
    if ref_name == 'uniform':
        ref_function = torch.ones(blocks_num, blocks_num) / blocks_num

    elif ref_name == 'linear':
        ref_function = torch.linspace(0, 1, blocks_num)

    elif ref_name == 'swa':
        ref_function = torch.zeros((blocks_num, blocks_num))
        row = torch.arange(blocks_num).unsqueeze(1)
        col = torch.arange(blocks_num).unsqueeze(0)
        mask = (row - col).abs() < window_size
        ref_function[mask] = float('inf')

    elif ref_name == 'clean':
        ref_function = torch.full((blocks_num, blocks_num), float('inf'))

    elif ref_name == 'interpolation':
        ref_function = torch.ones(blocks_num, blocks_num)
        idx = torch.arange(blocks_num)
        s = (idx.view(-1, 1) - idx.view(1, -1)).abs().float()

        # z 计算
        frac = torch.remainder(128.0 * s / 1560.0, 1.0)          # ∈ [0,1)
        z = (2.0 * (frac - 0.5).abs()) ** r                      # ∈ [0,1]

        # 衰减
        exp_a = torch.pow(a, 128.0 * s / 16380.0)
        exp_b = torch.pow(b, 128.0 * s / 16380.0)
        decay = k * (z * exp_a + (1.0 - z) * exp_b) + c

        mask_far = s > window_size
        ref_function[mask_far] *= decay[mask_far]                    # >128 ⇒ 衰减


    else:
        return  torch.ones(blocks_num, blocks_num) / blocks_num
        # raise ValueError(f"Invalid ref name: {ref_name}")

    return pad_ref_function(ref_function)

def pad_ref_function(ref_function):
    """pad trivial text part"""
    # from 1,1,1578,1578 to 1,1,1580,1580
    ref_function = ref_function.unsqueeze(0).unsqueeze(0)
    ref_function = F.pad(ref_function, (0, 2, 0, 2), mode='constant', value=float('inf'))
    return ref_function.squeeze(0).squeeze(0)

def parallelize_model(pipe, mesh, attention_type, attention_args, threshold_attn_args, sink_args, block_avg_args):
    """并行化模型"""
    parallelize_pipe(
        pipe,
        mesh=mesh,
        new_attention=jintao_sage,
        attention_args=attention_args,
        attention_type=attention_type,
        threshold_attn_args=threshold_attn_args,
        sink_args=sink_args,
        block_avg_args=block_avg_args,
    )
    parallelize_vae(pipe.vae, mesh=mesh._flatten())

def generate_video(pipe, prompt, height, width, num_frames, num_inference_steps, generator_seed):
    """生成视频"""
    output = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        output_type="pil" if dist.get_rank() == 0 else "pt",
        generator=torch.Generator(device="cuda").manual_seed(generator_seed),
    ).frames[0]
    
    return output

def save_video(output, output_dir, filename):
    """保存视频"""
    if dist.get_rank() == 0:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        export_to_video(output, os.path.join(output_dir, filename), fps=24)
        print(f"Video saved to: {os.path.join(output_dir, filename)}")

def get_filename(attention_type, params, num_frames):
    """生成文件名"""
    if attention_type == 'original':
        return f"hunyuan_video_original_{num_frames}.mp4"
    elif attention_type == 'XPOS':
        xpos = params.get('xpos', 0.95)
        return f"hunyuan_video_xpos_{xpos}_{num_frames}.mp4"
    elif attention_type == 'sigmoid':
        sigmoid_a = params.get('sigmoid_a', 1.0)
        return f"hunyuan_video_sigmoid_{sigmoid_a}_{num_frames}.mp4"
    elif attention_type == 'interpolation':
        alpha = params.get('alpha', 0.95)
        beta = params.get('beta', 0.90)
        return f"hunyuan_video_interpolation_{alpha}_{beta}_{num_frames}.mp4"
    elif attention_type == 'pattern':
        threshold = params.get('threshold', 0.5)
        xpos = params.get('xpos', 0.95)
        return f"hunyuan_video_pattern_{threshold}_{xpos}_{num_frames}.mp4"
    elif attention_type == 'repeat_interpolation':
        alpha = params.get('alpha', 0.95)
        beta = params.get('beta', 0.90)
        return f"hunyuan_video_repeat_interpolation_{alpha}_{beta}_{num_frames}.mp4"
    elif attention_type == 'sink':
        alpha = params.get('alpha', 1.00)
        beta = params.get('beta', 0.95)
        sink_width = params.get('sink_width', 4)
        window_width = params.get('window_width', 16)
        use_pattern_flags = params.get('use_pattern_flags', True)
        use_one_time_flags = params.get('use_one_time_flags', True)
        use_decay_mask = params.get('use_decay_mask', True)
        return f"hunyuan_video_sink_alpha_{alpha}_beta_{beta}_sinkwidth_{sink_width}_windowwidth_{window_width}_use_pattern_flags_{use_pattern_flags}_use_one_time_flags_{use_one_time_flags}_use_decay_mask_{use_decay_mask}_{num_frames}.mp4"
    elif attention_type == 'get_block_avg':
        return f"hunyuan_video_get_block_avg_{num_frames}.mp4"
    else:
        raise ValueError(f"Invalid attention type: {attention_type}")

def run_experiment(config, pipe):
    """运行单个实验"""
    # try:
    prompt_name = '_'.join(config['prompt'].split(' ')[0:10])
    filename = get_filename(config['attention_type'], config, config['num_frames'])
    if os.path.exists(f"output_videos_new/{prompt_name}/{config['attention_type']}/{filename}"):
        print(f"Video already exists: {filename}, skipping")
        return True
    
    # 设置参数
    attention_type = config['attention_type']
    num_frames = config['num_frames']
    height = config['height']
    width = config['width']
    prompt = config['prompt']
    num_inference_steps = config['num_inference_steps']
    generator_seed = config['generator_seed']
    
    # 计算frame_size
    frame_size = (height // 16) * (width // 16)
    
    # 获取attention masks
    attention_masks_narrow = None
    attention_masks_wide = None
    mask_selected_indices = None
    threshold_attn_args = None

    latents = (num_frames - 1) // 4 + 1

    if attention_type == 'pattern' or (attention_type == 'sink' and config['use_pattern_flags']):
        attention_masks_narrow, attention_masks_wide, mask_selected_indices = get_attention_masks(
            attention_type, 
            config['sample_mse_max_row'], 
            config['context_length'], 
            num_frames, 
            frame_size, 
            config['narrow_width'], 
            config['wide_width']
        )
        # print(f"attention_masks_narrow: {attention_masks_narrow.shape}")
        # print(f"attention_masks_wide: {attention_masks_wide.shape}")
        # 设置threshold参数
        threshold_attn_args = {
            'threshold': (config['narrow_width'] * 2 * config['pattern_threshold'] / ((num_frames - 1) // 4 + 1)),
            'pattern_threshold': config['pattern_threshold'],
            'one_time_ref': num_frames == 129,
            'mask_selected_indices': mask_selected_indices,
            'attention_masks_narrow': attention_masks_narrow,
            'attention_masks_wide': attention_masks_wide,
            'num_sampled_rows': config['sample_mse_max_row'],
            'sub_dir': prompt_name,
            'method': config['method'],
        }
    
    # 设置attention参数
    attention_args = {
        'frame_tokens': 2040,
        'prompt_name': prompt_name,
        'num_frames': num_frames,
        'use_pattern_flags': config['use_pattern_flags'] if attention_type not in ['get_block_avg', 'pattern'] else False,
        'use_one_time_flags': config['use_one_time_flags'] if attention_type not in ['get_block_avg', 'pattern'] else False,
        'use_decay_mask': config['use_decay_mask'] if attention_type not in ['get_block_avg', 'pattern'] else False,
        'use_one_timestep_block_avg': config['use_one_timestep_block_avg'] if attention_type not in ['get_block_avg', 'pattern'] else False,
        'use_block_bias': config['use_block_bias'] if attention_type not in ['get_block_avg', 'pattern'] else False,
    }

    if attention_args['use_decay_mask']:
        attention_args['decay_mask_threshold'] = config['decay_mask_threshold'] if attention_type not in ['get_block_avg', 'pattern'] else 0.0000
    
    # 根据attention_type添加特定参数
    if attention_type in ['XPOS']:
        xpos = config['xpos']
        attention_args.update({
            'xpos_xi': xpos**(1/16000),
        })
    
    if attention_type in ['interpolation', 'repeat_interpolation']:
        alpha = config['alpha']
        beta = config['beta']
        attention_args.update({
            'alpha_xpos_xi': float(alpha)**(1/16000),
            'beta_xpos_xi': float(beta)**(1/16000),
        })

    sink_args = None
    block_avg_args = {}
    if attention_type in ['sink']:
        sink_args = {
            'sink_width': config['sink_width'],
            'window_width': config['window_width'],
            'alpha_xpos_xi': float(config['alpha'])**(1/16000),
            'beta_xpos_xi': float(config['beta'])**(1/16000),
            'repeat_mask_in_sink': config['repeat_mask_in_sink'],
        }
        block_num = (latents * 2040 + 127) // 128
        if config['use_decay_mask']:
            block_avg_args.update({
                'threshold': config['decay_mask_threshold'],
            })

        if config['use_block_bias']:
            block_avg_args.update({
                'ref_function': get_ref_function(block_num, config['ref_function_name']),
            })


    if attention_type in ['get_block_avg']:
        attention_args.update({
            'block_size': 128,
            'prompt_name': prompt_name,
            'num_frames': num_frames,
            'world_size': dist.get_world_size(),
            'save_one_timestep': config['save_one_timestep'],
        })

    
    # 初始化mesh
    mesh = init_context_parallel_mesh(pipe.device.type)
    
    # 并行化模型
    parallelize_model(pipe, mesh, attention_type, attention_args, threshold_attn_args, sink_args, block_avg_args)
    
    # 启用VAE tiling
    pipe.vae.enable_tiling()
    
    # 生成视频
    output = generate_video(pipe, prompt, height, width, num_frames, num_inference_steps, generator_seed)
    
    # 保存视频
    prompt_name = '_'.join(prompt.split(' ')[0:10])
    output_dir = f"output_videos_new/{prompt_name}/{attention_type}"
    filename = get_filename(attention_type, config, num_frames)
    save_video(output, output_dir, filename)
    
    # 保存配置
    if dist.get_rank() == 0:
        config_file = os.path.join(output_dir, f"config_{filename.replace('.mp4', '.json')}")
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Config saved to: {config_file}")
    
    return True
        
    # except Exception as e:
    #     if dist.get_rank() == 0:
    #         print(f"Error in experiment: {e}")
    #     return False
    # finally:
    #     pass
    #     # dist.destroy_process_group()

def main():
    """主函数"""
    # 基础配置
    base_config = {
        'height': 544,
        'width': 960,
        'num_inference_steps': 1,
        'generator_seed': 42,
        'sample_mse_max_row': 64,
        'context_length': 256,
        'method': 'proportional',
        'wide_width': 2,
    }
    
    # 提示词列表
    prompts = [
        'An animated porcupine with a mix of brown and white fur and prominent quills is seen in a cozy, warmly lit interior setting, interacting with a green gift box with a yellow ribbon. The room is filled with wooden furniture and colorful wall decorations, suggesting a cheerful and domestic atmosphere. The porcupine\'s large eyes and expressive face convey a sense of lightheartedness and curiosity. The camera maintains a low angle, close to the ground, providing an intimate view of the character\'s actions without any movement, focusing on the playful and curious mood of the scene. The visual style is characteristic of contemporary 3D animation, with vibrant colors and smooth textures that create a polished and engaging look. The scene transitions to an outdoor environment, showcasing a sunny, verdant landscape with rocks, trees, and grass, indicating a natural, possibly forest-like setting. The presence of a small character in the final frame suggests the continuation of a narrative or the introduction of new characters.',
        'Animated characters, a rabbit and a mouse, are depicted in a perilous situation, first plummeting through a dark, undefined space, and then floating and swimming in a serene underwater environment. The characters are dressed in adventure gear, suggesting a narrative context. The camera closely follows their expressions and movements, capturing the tension and urgency of their situation. The medium and close-up shots emphasize their facial expressions, which convey fear and determination. The visual style is high-quality 3D animation with detailed textures and lighting, creating a cinematic feel.',
        'Drone footage captures the powerful waves crashing against the jagged cliffs of Portugal’s Algarve region. As the sun sets, it casts a fiery red hue across the rugged coastline, highlighting the dramatic rock formations and sea caves. Seabirds soar overhead, adding to the wild beauty of this picturesque landscape.',
        'Animated characters are engaging in a magical interaction within a dark, cavernous environment. The scene centers on a small, orange magical creature with a glowing heart, as well as two dragon-like creatures, one of which is holding a magical potion. The creature opens the potion, causing a transformation, which captures the attention of the dragons. Subsequently, two human characters with a torch discover the aftermath of the transformation, revealing a small, glowing creature resembling the one from earlier. The atmosphere is whimsical and magical, with a sense of curiosity and discovery. The camera remains static, offering medium shots that focus on the characters and their actions, while the visual style is traditional animation with smooth lines and vibrant colors.',
        # 'A man with facial hair, dressed in a burgundy shirt, is seen knocking on a weathered wooden door with a metal latch and a small window, set in a stone wall. The scene transitions to an indoor setting where the man, now wearing a blue shirt, speaks to the camera in a well-lit room furnished with a couch, a bookshelf, and various decorations. The video captures the man in a medium shot with a stationary camera, conveying a casual and friendly atmosphere in the indoor scene, contrasted with a neutral atmosphere in the outdoor scene. The visual style is realistic with natural lighting and color grading.'
    ]
    
    # 实验配置列表
    experiments = []
    
    # 1. Original attention
    # experiments.append({
    #     **base_config,
    #     'attention_type': 'original',
    #     'prompt': prompts[0],
    #     'num_frames': 129,
    # })
    
    # # 2. XPOS attention with different xpos values
    # xpos_values = [0.93, 0.95, 0.97, 0.98, 0.99]
    # for xpos in xpos_values:
    #     experiments.append({
    #         **base_config,
    #         'attention_type': 'XPOS',
    #         'xpos': xpos,
    #         'prompt': prompts[0],
    #         'num_frames': 393,
    #     })

    # # 3. Repeat XPOS 
    # for xpos in xpos_values:
    #     experiments.append({
    #         **base_config,
    #         'attention_type': 'repeat',
    #         'xpos': xpos,
    #         'prompt': prompts[0],
    #         'num_frames': 393,
    #     })
    
    # # 4. Sigmoid attention with different sigmoid_a values
    # sigmoid_values = [0.5, 1.0, 2.0]
    # for sigmoid_a in sigmoid_values:
    #     experiments.append({
    #         **base_config,
    #         'attention_type': 'sigmoid',
    #         'sigmoid_a': sigmoid_a,
    #         'prompt': prompts[0],
    #         'num_frames': 393,
    #     })
    
    # 5. Interpolation attention with different alpha/beta combinations
    alpha_beta_combinations = [
        # (0.90, 0.85),
        # (0.95, 0.90),
        # (0.99, 0.95),
        (1.00, 0.95),
        # (1.00, 0.90),
        # (1.00, 0.85),
    ]
    # for alpha, beta in alpha_beta_combinations:
    #     experiments.append({
    #         **base_config,
    #         'attention_type': 'interpolation',
    #         'alpha': alpha,
    #         'beta': beta,
    #         'prompt': prompts[0],
    #         'num_frames': 393,
    #     })
    
    # 6. Pattern attention with different thresholds
    
    # experiments.append({
    #     **base_config,
    #     'attention_type': 'pattern',
    #     'prompt': prompts[0],
    #     'num_frames': 129, # implying one_time_ref is True, cannot change

    #     # change below
    #     'pattern_threshold': 1,
    #     'narrow_width': 1,
    # })
    
    # # 7. get block avg 
    # experiments.append({
    #     **base_config,
    #     'attention_type': 'get_block_avg',
    #     'prompt': prompts[0],
    #     'num_frames': 393,

    #     'save_one_timestep': True,
    # })
    
    experiments.append({
        **base_config,
        'attention_type': 'sink',
        'prompt': prompts[0],
        'num_frames': 393,

        # --- decay part ---
        'alpha': 1.00,
        'beta': 0.95,

        # --- sink part ---
        'sink_width': 4,

        # --- repeat mask part ---
        'repeat_mask_in_sink': True,

        # --- sliding window part ---
        'window_width': 33,  # 这里是直径

        # --- online flags part ---
        'use_pattern_flags': True,
        'use_one_time_flags': True,
        'narrow_width': 1,
        'pattern_threshold': 1,  

        # --- block avg part ---
        'use_one_timestep_block_avg': True,

        # --- block bias part ---
        'use_block_bias': False,
        'ref_function_name': 'interpolation',

        # --- decay mask part ---
        'use_decay_mask': True,
        'decay_mask_threshold': 0.0001,
    })
    
    # 运行所有实验
    print(f"Total experiments to run: {len(experiments)}")

    # 初始化
    init_distributed()
    # pipe = load_model()
    
    for i, config in enumerate(experiments):
        print(f"\n{'='*50}")
        print(f"Running experiment {i+1}/{len(experiments)}")
        print(f"Attention type: {config['attention_type']}")
        print(f"Parameters: {config}")
        print(f"{'='*50}")
        
        # init_distributed()
        pipe = load_model()
        success = run_experiment(config, pipe)
        del pipe
        gc.collect()  # 强制垃圾回收
        
        if success:
            print(f"Experiment {i+1} completed successfully")
        else:
            print(f"Experiment {i+1} failed")
        
        # 添加一些延迟，避免GPU内存问题
        torch.cuda.empty_cache()
        import time
        time.sleep(5)

if __name__ == "__main__":
    main() 