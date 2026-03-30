import os
import shutil
import argparse
from pathlib import Path
import sys
import random
import json



def move_synthetic_data(synthetic_dir, target_dir, ipc: int = 3, training_ids=[]):  # ipc: numbers of images per camera per identity
    for cam in os.listdir(synthetic_dir):
        cam_name = cam[-1]
        for id in os.listdir(os.path.join(synthetic_dir, cam)):
            if id not in training_ids:
                continue
            real_id = str(int(id)+10000)
            images = [image for image in os.listdir(os.path.join(synthetic_dir, cam, id))]
            if len(images) < ipc:
                random_ipc_images = images
            else:
                random_ipc_images = random.sample(images, ipc)
            for image in random_ipc_images:
                new_name = str(f"{int(real_id):05d}") + f"_c{cam_name}" + "_" + image
                shutil.copy(os.path.join(synthetic_dir, cam, id, image), os.path.join(target_dir, new_name))



def main():
    parser = argparse.ArgumentParser(description='Move synthetic data to target directory')
    
    # Use more explicit argument names with flags
    parser.add_argument('--input', '-i', required=True,
                        help='Path to the input synthetic data directory')
    parser.add_argument('--output', '-o', required=True,
                        help='Path to the output target directory')
    
    args = parser.parse_args()
    
    dukemtmc_dir = "/data4/toan/projects/DiVE/dataset/reid_datasets/DukeMTMC/all_images/"
    dukemtmc_training_dir = '/data4/toan/projects/DiVE/dataset/NightReID_DukeMTMC_ori_stage1_envtoken_stage2_ipadapter_stage3_intensity_full_id/bounding_box_train'
    generated_img_dir = "/data4/toan/projects/DiVE/dataset/NightReID_DukeMTMC_ori_stage1_envtoken_stage2_ipadapter_stage3_intensity_full_id/output_stage3_step25000/cam1/"
    ids = []
    training_ids = []
    generated_ids = sorted([id for id in os.listdir(generated_img_dir)])
    images = sorted([image for image in os.listdir(dukemtmc_dir)])
    train_images = sorted([image for image in os.listdir(dukemtmc_training_dir)])
    for image in images:
        if image[:4] not in ids:
            ids.append(image[:4])
    for image in train_images:
        if image[:4] not in training_ids:
            training_ids.append(image[:4])
    mapping_dict = {ids[i] : generated_ids[i] for i in range(len(ids))}
    
    mapping_training_ids = [mapping_dict[id] for id in training_ids]
    
    
    print(len(mapping_training_ids))
    
    # move_synthetic_data(args.input, args.output, 3, mapping_training_ids)


if __name__ == "__main__":
    main()