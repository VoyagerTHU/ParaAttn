import torch
import torch.distributed as dist
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.utils import export_to_video
import sys
import os
import itertools
import json
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import para_attn
from jintao.core import jintao_sage
from para_attn.context_parallel import init_context_parallel_mesh
from para_attn.context_parallel.diffusers_adapters import parallelize_pipe
from para_attn.parallel_vae.diffusers_adapters import parallelize_vae

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
    if attention_type != 'pattern':
        return None, None, None
    
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
        
        if mask_narrow is not None:
            attention_masks_narrow.append(mask_narrow.to("cuda"))
        if mask_wide is not None:
            attention_masks_wide.append(mask_wide.to("cuda"))
    
    return attention_masks_narrow, attention_masks_wide, mask_selected_indices

def parallelize_model(pipe, mesh, attention_type, attention_args, threshold_attn_args):
    """并行化模型"""
    parallelize_pipe(
        pipe,
        mesh=mesh,
        new_attention=jintao_sage,
        attention_args=attention_args,
        attention_type=attention_type,
        method='thres',
        threshold_attn_args=threshold_attn_args,
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
    else:
        raise ValueError(f"Invalid attention type: {attention_type}")

def run_experiment(config):
    """运行单个实验"""
    try:
        # 初始化
        init_distributed()
        pipe = load_model()
        
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
        if attention_type in ['pattern']:
            attention_masks_narrow, attention_masks_wide, mask_selected_indices = get_attention_masks(
                attention_type, 
                config['sample_mse_max_row'], 
                config['context_length'], 
                num_frames, 
                frame_size, 
                config['narrow_width'], 
                config['wide_width']
            )
            # 设置threshold参数
            threshold_attn_args = {
                'threshold': config['threshold'],
                'one_time_ref': num_frames == 129,
                'mask_selected_indices': mask_selected_indices,
                'attention_masks_narrow': attention_masks_narrow,
                'attention_masks_wide': attention_masks_wide,
                'num_sampled_rows': config['sample_mse_max_row'],
                'sub_dir': '_'.join(prompt.split(' ')[0:3]),
                'cfg': True,
                'xi_for_XPOS': 0.9999934149894527,
            }
        
        # 设置attention参数
        attention_args = {
            'frame_tokens': 2040,
        }
        
        # 根据attention_type添加特定参数
        if attention_type in ['XPOS', 'pattern']:
            xpos = config['xpos']
            attention_args.update({
                'xpos_xi': xpos**(1/16000),
                'sigmoid_a': config['sigmoid_a'],
            })
        
        if attention_type in ['interpolation', 'repeat_interpolation']:
            alpha = config['alpha']
            beta = config['beta']
            attention_args.update({
                'alpha_xpos_xi': float(alpha)**(1/16000),
                'beta_xpos_xi': float(beta)**(1/16000),
            })
        
        
        # 初始化mesh
        mesh = init_context_parallel_mesh(pipe.device.type)
        
        # 并行化模型
        parallelize_model(pipe, mesh, attention_type, attention_args, threshold_attn_args)
        
        # 启用VAE tiling
        pipe.vae.enable_tiling()
        
        # 生成视频
        output = generate_video(pipe, prompt, height, width, num_frames, num_inference_steps, generator_seed)
        
        # 保存视频
        prompt_name = '_'.join(prompt.split(' ')[0:3])
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
        
    except Exception as e:
        if dist.get_rank() == 0:
            print(f"Error in experiment: {e}")
        return False
    finally:
        dist.destroy_process_group()

def main():
    """主函数"""
    # 基础配置
    base_config = {
        'height': 544,
        'width': 960,
        'num_inference_steps': 50,
        'generator_seed': 42,
        'sample_mse_max_row': 64,
        'context_length': 256,
        'narrow_width': 1,
        'wide_width': 2,
        'sigmoid_a': 1.0,
    }
    
    # 提示词列表
    prompts = [
        'An animated porcupine with a mix of brown and white fur and prominent quills is seen in a cozy, warmly lit interior setting, interacting with a green gift box with a yellow ribbon. The room is filled with wooden furniture and colorful wall decorations, suggesting a cheerful and domestic atmosphere. The porcupine\'s large eyes and expressive face convey a sense of lightheartedness and curiosity. The camera maintains a low angle, close to the ground, providing an intimate view of the character\'s actions without any movement, focusing on the playful and curious mood of the scene. The visual style is characteristic of contemporary 3D animation, with vibrant colors and smooth textures that create a polished and engaging look. The scene transitions to an outdoor environment, showcasing a sunny, verdant landscape with rocks, trees, and grass, indicating a natural, possibly forest-like setting. The presence of a small character in the final frame suggests the continuation of a narrative or the introduction of new characters.',
        'Animated characters, a rabbit and a mouse, are depicted in a perilous situation, first plummeting through a dark, undefined space, and then floating and swimming in a serene underwater environment. The characters are dressed in adventure gear, suggesting a narrative context. The camera closely follows their expressions and movements, capturing the tension and urgency of their situation. The medium and close-up shots emphasize their facial expressions, which convey fear and determination. The visual style is high-quality 3D animation with detailed textures and lighting, creating a cinematic feel.',
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
        (0.90, 0.85),
        (0.95, 0.90),
        (0.99, 0.95),
        (1.00, 0.95),
        (1.00, 0.90),
        (1.00, 0.85),
    ]
    for alpha, beta in alpha_beta_combinations:
        experiments.append({
            **base_config,
            'attention_type': 'interpolation',
            'alpha': alpha,
            'beta': beta,
            'prompt': prompts[0],
            'num_frames': 129,
        })
    
    # # 6. Pattern attention with different thresholds
    # threshold_values = [0.3, 0.5, 0.7]
    # for threshold in threshold_values:
    #     experiments.append({
    #         **base_config,
    #         'attention_type': 'pattern',
    #         'threshold': threshold,
    #         'xpos': 0.95,
    #         'prompt': prompts[0],
    #         'num_frames': 393,
    #     })
    
    # 7. Repeat interpolation with different alpha/beta combinations
    for alpha, beta in alpha_beta_combinations:
        experiments.append({
            **base_config,
            'attention_type': 'repeat_interpolation',
            'alpha': alpha,
            'beta': beta,
            'prompt': prompts[0],
            'num_frames': 393,
        })
    
    # 运行所有实验
    print(f"Total experiments to run: {len(experiments)}")
    
    for i, config in enumerate(experiments):
        print(f"\n{'='*50}")
        print(f"Running experiment {i+1}/{len(experiments)}")
        print(f"Attention type: {config['attention_type']}")
        print(f"Parameters: {config}")
        print(f"{'='*50}")
        
        success = run_experiment(config)
        
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