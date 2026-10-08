#!/bin/bash
# sample.sh - Script for generating synthetic images using generate_t2i.py

function sample {
    # Default arguments if not provided
    data_dir=${1:-"/data4/toan/projects/DNA-Diff/dataset/NightReID_Market1501/"}
    cond_daytime_path=${2:-"${data_dir%/}/cam7/env61/"}
    model_path=${3:-"/data4/toan/projects/DiVE/stable-diffusion-v1-5"}
    output_dir=${4:-"${data_dir%/}/finetune_model/finetune_stage3_ipadapter/output_step30000_test"}
    num_env_tokens_per_camera=${5:-10}
    resolution=${6:-512}
    use_pretrained_weights=${7:-false}
    
    echo "Starting image generation with:"
    echo "- Data directory: $data_dir"
    echo "- Condition daytime images path: $cond_daytime_path"
    echo "- Model path: $model_path"
    echo "- LoRA path: $lora_path" 
    echo "- Embedding path: $embed_path"
    echo "- Output directory: $output_dir"
    echo "- Number of environment tokens per camera: $num_env_tokens_per_camera"
    echo "- Resolution: $resolution"
    
    # Export the data directory for the Python script to use
    export HUG_LOCAL_IMAGE_TRAIN_DIR="$data_dir"
    
    python generate_t2i.py \
    --output_path "$output_dir" \
    --cond_daytime_path "$cond_daytime_path" \
    --num_env_tokens_per_camera "$num_env_tokens_per_camera" \
    --resolution $resolution \
    --gpu_ids 1 2 3  \
    --cameras cam1 cam2 cam3 cam4 cam5 cam6 \
    --repeat_each_id 3 \
    --batch_size 60 \
    --images_per_call 1 \
    --model_path "$model_path" \
    --lora_path "$lora_path" \
    --embed_path "$embed_path" \
    --guidance_scale 7.5 \
    --use_pretrained_weights "$use_pretrained_weights"
    # Uncomment the line below to run in debug mode (single-process)
    # --debug
}

# To run this script: source script/sample.sh && CUDA_VISIBLE_DEVICES=0,1,2 sample
#      
# NOTE ON PARAMETERS:
# 1. data_dir: Path to the training/input data directory (sets HUG_LOCAL_IMAGE_TRAIN_DIR)
# 2. cond_daytime_path: Path to daytime training dataset (Market1501 dataset)
# 3. model_path: Path to the base Stable Diffusion model
# 4. output_dir: Directory to save generated images
# 5. num_env_tokens_per_camera: number of environment per camera (default: 10)
# 6. resolution: Image size in pixels (default: 512)
# 7. use_pretrained_weights: Whether to use pretrained weights for generation (default: false)

# If this script is called directly (not sourced), run the sample function
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    sample "$@"
fi