#!/bin/bash

# HunyuanVideo Low VRAM Multi-GPU Script
# Optimized for GPUs with limited memory (each GPU needs ~15-20GB)

# Set environment variables for memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0

# Memory management
export PYTORCH_NO_CUDA_MEMORY_CACHING=0

echo "Starting HunyuanVideo Low VRAM Multi-GPU Inference..."
echo "==================================================="

# Check GPU count and memory
echo "GPU Information:"
nvidia-smi --list-gpus
echo ""

# Determine number of GPUs to use (use 4 or 6 based on available)
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
if [ $GPU_COUNT -ge 6 ]; then
    NPROC=6
elif [ $GPU_COUNT -ge 4 ]; then
    NPROC=4
else
    NPROC=2
fi

echo "Using $NPROC GPUs for parallel inference"
echo ""

# Run with torchrun
torchrun --nproc_per_node=$NPROC \
    --master_port=29500 \
    parallel_examples/run_hunyuan_video_low_vram.py

echo ""
echo "HunyuanVideo inference completed!"
echo "Check output: hunyuan_video_low_vram.mp4" 