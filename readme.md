# DNA-Diff

## Environment Setup

1. Clone this repo

2. Create a conda environment using the provided environment.yml file

3. Activate the created environment

## Preprocess

1. Download the NightReID and Market1501 dataset

2. Separate each camera images into 10 different environmental groups by intensity
   Run split_by_lighting.py file

3. Merge Market1501 dataset with NightReID dataset
   Run Preprocess/market2nightreid.py

4. Convert the NightReID + Market1501 dataset into a format ready for training
   Run Preprocess/c2hf.py --dataset NightReID_Market1501 --base_path path/to/nightreid_market1501 --num_cameras 7

## Data Generation

1. Download SD-1.5 model

2. Finetune stage 1
   cd Data_Generation
   source script/finetune_stage1.sh
   export CUDA_VISIBLE_DEVICES=1,2,3
   finetune

3. Finetune stage 2
   cd Data_Generation
   source script/finetune_stage2.sh
   export GPU_IDS=(0,1,2)
   finetune

4. Finetune stage 3
   cd Data_Generation
   source script/finetune_stage3.sh
   export GPU_IDS=(0,1,2)
   finetune

5. Finetune all stages

6. Inference
