#!/bin/bash

GPU_IDS=1,2,3,4,5,6,7
# GPU_IDS=7


echo "===== Step 1 ====="
(
    source scripts/finetune_step1.sh
    CUDA_VISIBLE_DEVICES=$GPU_IDS finetune
) && \

echo "===== Step 2 ====="
(
    source scripts/finetune_step2.sh
    CUDA_VISIBLE_DEVICES=$GPU_IDS finetune
) && \

echo "===== Step 3 ====="
(
    source scripts/finetune_step3.sh
    CUDA_VISIBLE_DEVICES=$GPU_IDS finetune
)

echo "===== All steps completed ====="
