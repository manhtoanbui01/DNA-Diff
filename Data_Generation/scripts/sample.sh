#!/bin/bash
# sample.sh - Script for generating synthetic images using generate_t2i.py

function sample {
    # Default arguments if not provided
    data_dir=${1:-"../dataset/NightReID_Market1501/"}
    cond_daytime_path=${2:-"${data_dir%/}/cam7/env61/"}
    model_path=${3:-"../stable-diffusion-v1-5"}
    output_dir=${4:-"${data_dir%/}/finetune_model/finetune_stage3_ipadapter/output"}
    num_env_tokens_per_camera=${5:-10}
    resolution=${6:-512}
    use_pretrained_weights=${7:-false}

    # Previously hardcoded inside the python call - now overridable, by position
    # or by environment variable, the same way the finetune_step*.sh scripts work.
    gpu_ids=${8:-${GPU_IDS:-"1 2 3"}}
    cameras=${9:-${CAMERAS:-"cam1 cam2 cam3 cam4 cam5 cam6"}}
    repeat_each_id=${10:-${REPEAT_EACH_ID:-3}}
    batch_size=${11:-${BATCH_SIZE:-60}}
    images_per_call=${12:-${IMAGES_PER_CALL:-1}}
    guidance_scale=${13:-${GUIDANCE_SCALE:-7.5}}

    # Accept "1,2,3" as well as "1 2 3" so GPU_IDS can be shared with the
    # finetune scripts, which pass it to `accelerate --gpu_ids` comma-separated.
    gpu_ids=$(echo "$gpu_ids" | tr ',' ' ')
    cameras=$(echo "$cameras" | tr ',' ' ')

    echo "Starting image generation with:"
    echo "- Data directory: $data_dir"
    echo "- Condition daytime images path: $cond_daytime_path"
    echo "- Model path: $model_path"
    echo "- Output directory: $output_dir"
    echo "- Number of environment tokens per camera: $num_env_tokens_per_camera"
    echo "- Resolution: $resolution"
    echo "- GPU IDs: $gpu_ids"
    echo "- Cameras: $cameras"
    echo "- Repeat each ID: $repeat_each_id"
    echo "- Batch size: $batch_size"
    echo "- Images per call: $images_per_call"
    echo "- Guidance scale: $guidance_scale"
    echo "- Use pretrained weights: $use_pretrained_weights"
    echo "- Debug (single-process): ${DEBUG:-0}"

    # Export the data directory for the Python script to use
    export HUG_LOCAL_IMAGE_TRAIN_DIR="$data_dir"

    # DEBUG=1 runs generation in a single process, which makes tracebacks readable.
    debug_arg=""
    if [ "${DEBUG:-0}" != "0" ]; then
        debug_arg="--debug"
    fi

    # $gpu_ids and $cameras are deliberately unquoted: --gpu_ids and --cameras
    # take nargs="+", so the shell must split them into separate arguments.
    python generate_t2i.py \
    --output_path "$output_dir" \
    --cond_daytime_path "$cond_daytime_path" \
    --num_env_tokens_per_camera "$num_env_tokens_per_camera" \
    --resolution "$resolution" \
    --gpu_ids $gpu_ids \
    --cameras $cameras \
    --repeat_each_id "$repeat_each_id" \
    --batch_size "$batch_size" \
    --images_per_call "$images_per_call" \
    --model_path "$model_path" \
    --guidance_scale "$guidance_scale" \
    --use_pretrained_weights "$use_pretrained_weights" \
    $debug_arg
}

# To run this script:
#
# 1. Basic usage with default settings:
#    source scripts/sample.sh
#    sample
#
# 2. Specify data directory and model:
#    sample "/path/to/data" "" "/path/to/model"
#    (pass "" for any argument you want to leave at its default)
#
# 3. Override the generation settings by environment variable:
#    GPU_IDS="0,1" CAMERAS="cam1 cam2" BATCH_SIZE=32 sample
#
# 4. Run single-process for a readable traceback:
#    DEBUG=1 sample
#
# NOTE ON PARAMETERS:
#  1. data_dir:                  Input data directory (sets HUG_LOCAL_IMAGE_TRAIN_DIR)
#  2. cond_daytime_path:         Daytime reference images (the Market-1501 camera)
#  3. model_path:                Path to the base Stable Diffusion model
#  4. output_dir:                Directory to save generated images
#  5. num_env_tokens_per_camera: Number of environments per camera (default: 10)
#  6. resolution:                Image size in pixels (default: 512)
#  7. use_pretrained_weights:    Use pretrained weights for generation (default: false)
#  8. gpu_ids:                   GPUs to use, one worker each    (env GPU_IDS,         default "1 2 3")
#  9. cameras:                   Target night-time cameras       (env CAMERAS,         default cam1..cam6)
# 10. repeat_each_id:            Times each ID is enqueued/camera (env REPEAT_EACH_ID,  default 3)
# 11. batch_size:                Queue items per pipeline call   (env BATCH_SIZE,      default 60)
# 12. images_per_call:           Images per pipeline call        (env IMAGES_PER_CALL, default 1)
# 13. guidance_scale:            Classifier-free guidance scale  (env GUIDANCE_SCALE,  default 7.5)
#
# Also honoured: DEBUG=1 to run single-process.
#
# Positional arguments take precedence over the environment variables.
# Do NOT also set CUDA_VISIBLE_DEVICES - gpu_ids are physical indices and
# setting both remaps them twice.

# If this script is called directly (not sourced), run the sample function
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    sample "$@"
fi
