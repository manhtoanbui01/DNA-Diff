#!/bin/bash

GPU_IDS=1,2,3
# GPU_IDS=0


# echo "===== Step 1 ====="
# (
#     source scripts/finetune_step1.sh
#     CUDA_VISIBLE_DEVICES=$GPU_IDS finetune
# ) && \

# echo "===== Step 2 ====="
# (
#     source scripts/finetune_step2.sh
#     CUDA_VISIBLE_DEVICES=$GPU_IDS finetune
# ) && \

echo "===== Step 3 ====="
(
    source scripts/finetune_step3.sh
    CUDA_VISIBLE_DEVICES=$GPU_IDS finetune)
# ) && \

# echo "===== Inference ====="
# (
#     source scripts/sample.sh
#     CUDA_VISIBLE_DEVICES=$GPU_IDS sample
# )

# echo "===== All steps completed ====="
