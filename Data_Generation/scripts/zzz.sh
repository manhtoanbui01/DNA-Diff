#!/bin/bash

# sleep 3h

# source scripts/sample.sh
# sample

# cd ..

# git add .
# git commit -m "." || true

# git checkout ori_step1_envtoken_step2_ipadapter_step3_full_12_cameras

# cd Data_Generation

# source scripts/finetune_step1.sh
# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 finetune

source scripts/finetune_step2.sh
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 finetune

source scripts/finetune_step3.sh
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 finetune