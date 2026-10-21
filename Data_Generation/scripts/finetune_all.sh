#!/bin/bash
# finetune_all.sh - Run the full 3-stage finetuning pipeline sequentially
#
#   Step 1: train_t2i_ila.py       -> finetune_model/finetune_stage1_ila/
#   Step 2: train_t2i_envtoken.py   -> finetune_model/finetune_stage2_envtoken/
#   Step 3: train_t2i_ipadapter.py  -> finetune_model/finetune_stage3_ipadapter/
#
# Each stage loads the previous stage's weights. Those paths are HARDCODED inside
# the training scripts and are keyed to the stage's max_train_steps:
#   train_t2i_envtoken.py  reads  finetune_stage1_ila/checkpoint-100000/model.safetensors
#                          and    finetune_stage1_ila/learned_embeds-steps-100000.bin
#   train_t2i_ipadapter.py reads  finetune_stage2_envtoken/checkpoint-40000/model.safetensors
#                          and    finetune_stage2_envtoken/learned_embeds-steps-40000.bin
# So if you change STEP1_MAX_STEPS / STEP2_MAX_STEPS you must also update those
# paths in the training scripts (this script warns when they diverge).

# Resolve locations so the script can be launched from anywhere.
# The training scripts are referenced by relative name, so we run from Data_Generation/.
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$(dirname "$SCRIPTS_DIR")"

function finetune_all {
    # Default arguments if not provided
    data_dir=${1:-"/data4/toan/projects/DNA-Diff/dataset/NightReID_Market1501/"}
    model_path=${2:-"/data4/toan/projects/DiVE/stable-diffusion-v1-5"}
    steps=${3:-${STEPS:-"1 2 3"}}
    batchsize=${4:-32}
    nepoch=${5:-100}
    rank=${6:-128}

    # Get GPU IDs - allow it to be passed as 7th parameter or use environment variable
    gpu_ids=${7:-${GPU_IDS:-1,2,3}}
    # NOTE: gpu_ids are physical GPU indices handed to `accelerate --gpu_ids`.
    # Do NOT also set CUDA_VISIBLE_DEVICES, or the indices get remapped twice.

    # Per-stage schedules (see the header note before changing these)
    step1_max_steps=${STEP1_MAX_STEPS:-100000}
    step1_ckpt_steps=${STEP1_CKPT_STEPS:-25000}
    step2_max_steps=${STEP2_MAX_STEPS:-40000}
    step2_ckpt_steps=${STEP2_CKPT_STEPS:-10000}
    step3_max_steps=${STEP3_MAX_STEPS:-30000}
    step3_ckpt_steps=${STEP3_CKPT_STEPS:-10000}

    model_root="${data_dir%/}/finetune_model"
    stage1_out="$model_root/finetune_stage1_ila/"
    stage2_out="$model_root/finetune_stage2_envtoken/"
    stage3_out="$model_root/finetune_stage3_ipadapter/"

    log_dir=${LOG_DIR:-"$model_root/logs"}
    run_stamp=$(date +%Y%m%d_%H%M%S)

    echo "================ Full finetuning pipeline ================"
    echo "- Data directory:   $data_dir"
    echo "- Model path:       $model_path"
    echo "- Steps to run:     $steps"
    echo "- Batch size:       $batchsize"
    echo "- Number of epochs: $nepoch"
    echo "- LoRA rank:        $rank"
    echo "- GPU IDs:          $gpu_ids"
    echo "- Working dir:      $WORK_DIR"
    echo "- Log directory:    $log_dir"
    echo "=========================================================="

    if [ ! -d "$data_dir" ]; then
        echo "ERROR: data directory does not exist: $data_dir"
        return 1
    fi
    if [ ! -d "$model_path" ]; then
        echo "ERROR: base model directory does not exist: $model_path"
        return 1
    fi
    mkdir -p "$log_dir" || return 1

    # Warn if the stage schedules no longer match the hardcoded cross-stage paths
    if ! grep -q "checkpoint-${step1_max_steps}" "$WORK_DIR/train_t2i_envtoken.py"; then
        echo "WARNING: train_t2i_envtoken.py does not reference checkpoint-${step1_max_steps};"
        echo "         step 2 will not pick up the step 1 weights produced by this run."
    fi
    if ! grep -q "checkpoint-${step2_max_steps}" "$WORK_DIR/train_t2i_ipadapter.py"; then
        echo "WARNING: train_t2i_ipadapter.py does not reference checkpoint-${step2_max_steps};"
        echo "         step 3 will not pick up the step 2 weights produced by this run."
    fi

    # Verify a finished stage left behind the artifacts the next stage needs
    function check_stage_output {
        stage_name=$1
        stage_dir=$2
        stage_steps=$3

        ckpt="${stage_dir%/}/checkpoint-${stage_steps}/model.safetensors"
        embeds="${stage_dir%/}/learned_embeds-steps-${stage_steps}.bin"
        missing=0
        for f in "$ckpt" "$embeds"; do
            if [ ! -f "$f" ]; then
                echo "ERROR: $stage_name output missing: $f"
                missing=1
            fi
        done
        return $missing
    }

    # Run one stage in a subshell so the per-step `finetune` definitions don't collide
    function run_stage {
        step_no=$1
        step_script=$2
        step_out=$3
        step_max_steps=$4
        step_ckpt_steps=$5

        log_file="$log_dir/step${step_no}_${run_stamp}.log"
        echo ""
        echo "===== Step $step_no ($step_script) ====="
        echo "Logging to: $log_file"

        (
            cd "$WORK_DIR" || exit 1
            source "$SCRIPTS_DIR/$step_script" || exit 1
            finetune "$data_dir" "$model_path" "$step_out" \
                     "$step_max_steps" "$batchsize" "$nepoch" "$rank" \
                     "$step_ckpt_steps" "$gpu_ids"
        ) 2>&1 | tee "$log_file"

        status=${PIPESTATUS[0]}
        if [ "$status" -ne 0 ]; then
            echo "ERROR: step $step_no failed with exit code $status (see $log_file)"
            return "$status"
        fi
        echo "===== Step $step_no finished ====="
        return 0
    }

    for step in $steps; do
        case "$step" in
            1)
                run_stage 1 "finetune_step1.sh" "$stage1_out" \
                          "$step1_max_steps" "$step1_ckpt_steps" || return 1
                check_stage_output "Step 1" "$stage1_out" "$step1_max_steps" || return 1
                ;;
            2)
                # Step 2 consumes step 1's LoRA + learned embeddings
                check_stage_output "Step 1" "$stage1_out" "$step1_max_steps" || {
                    echo "       Run step 1 first, or add it to STEPS."
                    return 1
                }
                run_stage 2 "finetune_step2.sh" "$stage2_out" \
                          "$step2_max_steps" "$step2_ckpt_steps" || return 1
                check_stage_output "Step 2" "$stage2_out" "$step2_max_steps" || return 1
                ;;
            3)
                # Step 3 consumes step 1's embeddings and step 2's env-token weights
                check_stage_output "Step 1" "$stage1_out" "$step1_max_steps" || {
                    echo "       Run step 1 first, or add it to STEPS."
                    return 1
                }
                check_stage_output "Step 2" "$stage2_out" "$step2_max_steps" || {
                    echo "       Run step 2 first, or add it to STEPS."
                    return 1
                }
                run_stage 3 "finetune_step3.sh" "$stage3_out" \
                          "$step3_max_steps" "$step3_ckpt_steps" || return 1
                ;;
            *)
                echo "ERROR: unknown step '$step' (expected 1, 2 or 3)"
                return 1
                ;;
        esac
    done

    echo ""
    echo "===== All requested steps completed ($steps) ====="
    echo "Stage 1 output: $stage1_out"
    echo "Stage 2 output: $stage2_out"
    echo "Stage 3 output: $stage3_out"
    echo "Next: run scripts/sample.sh to generate images with the stage 3 model."
    return 0
}

# To run this script:
#
# 1. Basic usage with default settings (runs steps 1 -> 2 -> 3):
#    bash scripts/finetune_all.sh
#
# 2. Specify data directory and model:
#    bash scripts/finetune_all.sh "/path/to/data" "/path/to/model"
#
# 3. Resume from a later step (previous stage outputs must already exist):
#    STEPS="2 3" bash scripts/finetune_all.sh
#    bash scripts/finetune_all.sh "/path/to/data" "/path/to/model" "3"
#
# 4. Set GPU IDs via parameter (7th position) or environment variable:
#    bash scripts/finetune_all.sh "/path/to/data" "/path/to/model" "1 2 3" 32 100 128 "0,1"
#    GPU_IDS="0,1" bash scripts/finetune_all.sh
#
# 5. Override the per-stage schedules (remember to update the hardcoded
#    cross-stage checkpoint paths in the training scripts as well):
#    STEP1_MAX_STEPS=50000 STEP2_MAX_STEPS=20000 STEP3_MAX_STEPS=15000 \
#        bash scripts/finetune_all.sh
#
# Parameter explanations:
# 1. data_dir:   Path to the training/input data directory
# 2. model_path: Path to the base Stable Diffusion model
# 3. steps:      Which stages to run, e.g. "1 2 3", "2 3", "3"
# 4. batchsize:  Number of images per batch
# 5. nepoch:     Number of training epochs
# 6. rank:       LoRA rank (higher = more capacity but uses more memory)
# 7. gpu_ids:    List of GPU IDs to use, e.g. "1,2,3" or "1 2 3"
#
# Environment variables:
#   STEPS, GPU_IDS, LOG_DIR,
#   STEP1_MAX_STEPS, STEP1_CKPT_STEPS,
#   STEP2_MAX_STEPS, STEP2_CKPT_STEPS,
#   STEP3_MAX_STEPS, STEP3_CKPT_STEPS

# If this script is called directly (not sourced), run the finetune_all function
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    finetune_all "$@"
fi
