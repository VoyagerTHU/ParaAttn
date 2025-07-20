# 实验配置文件
# 你可以在这里定义不同的实验参数组合

# 基础配置
BASE_CONFIG = {
    'height': 544,
    'width': 960,
    'num_inference_steps': 30,
    'generator_seed': 42,
    'sample_mse_max_row': 64,
    'context_length': 256,
    'narrow_width': 1,
    'wide_width': 2,
    'sigmoid_a': 1.0,
}

# 提示词列表
PROMPTS = [
    'An animated porcupine with a mix of brown and white fur and prominent quills is seen in a cozy, warmly lit interior setting, interacting with a green gift box with a yellow ribbon. The room is filled with wooden furniture and colorful wall decorations, suggesting a cheerful and domestic atmosphere. The porcupine\'s large eyes and expressive face convey a sense of lightheartedness and curiosity. The camera maintains a low angle, close to the ground, providing an intimate view of the character\'s actions without any movement, focusing on the playful and curious mood of the scene. The visual style is characteristic of contemporary 3D animation, with vibrant colors and smooth textures that create a polished and engaging look. The scene transitions to an outdoor environment, showcasing a sunny, verdant landscape with rocks, trees, and grass, indicating a natural, possibly forest-like setting. The presence of a small character in the final frame suggests the continuation of a narrative or the introduction of new characters.',
    'Animated characters, a rabbit and a mouse, are depicted in a perilous situation, first plummeting through a dark, undefined space, and then floating and swimming in a serene underwater environment. The characters are dressed in adventure gear, suggesting a narrative context. The camera closely follows their expressions and movements, capturing the tension and urgency of their situation. The medium and close-up shots emphasize their facial expressions, which convey fear and determination. The visual style is high-quality 3D animation with detailed textures and lighting, creating a cinematic feel.',
    'A man with facial hair, dressed in a burgundy shirt, is seen knocking on a weathered wooden door with a metal latch and a small window, set in a stone wall. The scene transitions to an indoor setting where the man, now wearing a blue shirt, speaks to the camera in a well-lit room furnished with a couch, a bookshelf, and various decorations. The video captures the man in a medium shot with a stationary camera, conveying a casual and friendly atmosphere in the indoor scene, contrasted with a neutral atmosphere in the outdoor scene. The visual style is realistic with natural lighting and color grading.',
]

# 实验配置定义
EXPERIMENT_CONFIGS = {
    # 1. 原始attention作为baseline
    'original_baseline': {
        'attention_type': 'original',
        'num_frames': 129,
        'prompt': PROMPTS[0],
    },
    
    # 2. XPOS attention实验
    'xpos_experiments': [
        {
            'attention_type': 'XPOS',
            'xpos': xpos,
            'num_frames': 129,
            'prompt': PROMPTS[0],
        }
        for xpos in [0.85, 0.90, 0.95, 0.99]
    ],
    
    # 3. Sigmoid attention实验
    'sigmoid_experiments': [
        {
            'attention_type': 'sigmoid',
            'sigmoid_a': sigmoid_a,
            'num_frames': 129,
            'prompt': PROMPTS[0],
        }
        for sigmoid_a in [0.5, 1.0, 1.5, 2.0]
    ],
    
    # 4. Interpolation attention实验
    'interpolation_experiments': [
        {
            'attention_type': 'interpolation',
            'alpha': alpha,
            'beta': beta,
            'num_frames': 129,
            'prompt': PROMPTS[0],
        }
        for alpha, beta in [
            (0.85, 0.80),
            (0.90, 0.85),
            (0.95, 0.90),
            (0.99, 0.95),
        ]
    ],
    
    # 5. Pattern attention实验
    'pattern_experiments': [
        {
            'attention_type': 'pattern',
            'threshold': threshold,
            'xpos': 0.95,
            'num_frames': 129,
            'prompt': PROMPTS[0],
        }
        for threshold in [0.2, 0.3, 0.5, 0.7, 0.8]
    ],
    
    # 6. Repeat interpolation实验
    'repeat_interpolation_experiments': [
        {
            'attention_type': 'repeat_interpolation',
            'alpha': alpha,
            'beta': beta,
            'num_frames': 129,
            'prompt': PROMPTS[0],
        }
        for alpha, beta in [
            (0.85, 0.80),
            (0.90, 0.85),
            (0.95, 0.90),
            (0.99, 0.95),
        ]
    ],
    
    # 7. 长视频实验 (393帧)
    'long_video_experiments': [
        {
            'attention_type': 'original',
            'num_frames': 393,
            'prompt': PROMPTS[1],
        },
        {
            'attention_type': 'XPOS',
            'xpos': 0.95,
            'num_frames': 393,
            'prompt': PROMPTS[1],
        },
        {
            'attention_type': 'pattern',
            'threshold': 0.5,
            'xpos': 0.95,
            'num_frames': 393,
            'prompt': PROMPTS[1],
        },
        {
            'attention_type': 'repeat_interpolation',
            'alpha': 0.95,
            'beta': 0.90,
            'num_frames': 393,
            'prompt': PROMPTS[1],
        },
    ],
    
    # 8. 多提示词对比实验
    'multi_prompt_comparison': [
        {
            'attention_type': 'original',
            'num_frames': 129,
            'prompt': prompt,
        }
        for prompt in PROMPTS
    ] + [
        {
            'attention_type': 'XPOS',
            'xpos': 0.95,
            'num_frames': 129,
            'prompt': prompt,
        }
        for prompt in PROMPTS
    ],
    
    # 9. 参数敏感性分析
    'sensitivity_analysis': [
        # XPOS参数敏感性
        {
            'attention_type': 'XPOS',
            'xpos': xpos,
            'num_frames': 129,
            'prompt': PROMPTS[0],
        }
        for xpos in [0.80, 0.85, 0.90, 0.92, 0.94, 0.95, 0.96, 0.98, 0.99, 0.999]
    ] + [
        # Pattern阈值敏感性
        {
            'attention_type': 'pattern',
            'threshold': threshold,
            'xpos': 0.95,
            'num_frames': 129,
            'prompt': PROMPTS[0],
        }
        for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ],
}

# 预定义的实验组合
EXPERIMENT_SETS = {
    'quick_test': [
        'original_baseline',
        'xpos_experiments',
        'pattern_experiments',
    ],
    
    'comprehensive': [
        'original_baseline',
        'xpos_experiments',
        'sigmoid_experiments',
        'interpolation_experiments',
        'pattern_experiments',
        'repeat_interpolation_experiments',
        'long_video_experiments',
    ],
    
    'sensitivity': [
        'sensitivity_analysis',
    ],
    
    'multi_prompt': [
        'multi_prompt_comparison',
    ],
    
    'all': list(EXPERIMENT_CONFIGS.keys()),
}

def get_experiments(experiment_set='comprehensive'):
    """
    获取指定实验集合的配置
    
    Args:
        experiment_set: 实验集合名称，可以是预定义的集合或具体的实验名称列表
    
    Returns:
        list: 实验配置列表
    """
    if experiment_set in EXPERIMENT_SETS:
        experiment_names = EXPERIMENT_SETS[experiment_set]
    else:
        experiment_names = [experiment_set]
    
    experiments = []
    
    for name in experiment_names:
        if name in EXPERIMENT_CONFIGS:
            config = EXPERIMENT_CONFIGS[name]
            if isinstance(config, list):
                experiments.extend(config)
            else:
                experiments.append(config)
    
    # 合并基础配置
    final_experiments = []
    for exp in experiments:
        final_experiments.append({**BASE_CONFIG, **exp})
    
    return final_experiments

def print_experiment_summary(experiment_set='comprehensive'):
    """打印实验摘要"""
    experiments = get_experiments(experiment_set)
    
    print(f"实验集合: {experiment_set}")
    print(f"总实验数: {len(experiments)}")
    print("\n实验详情:")
    
    attention_types = {}
    for i, exp in enumerate(experiments):
        attn_type = exp['attention_type']
        if attn_type not in attention_types:
            attention_types[attn_type] = []
        attention_types[attn_type].append(i + 1)
    
    for attn_type, indices in attention_types.items():
        print(f"  {attn_type}: 实验 {indices}")
    
    return experiments 