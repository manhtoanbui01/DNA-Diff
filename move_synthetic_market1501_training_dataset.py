import os
import shutil
import argparse
from pathlib import Path
import sys
import random
import json



def move_synthetic_data(synthetic_dir, target_dir, ipc: int = 3, training_ids=[]):  # ipc: numbers of images per camera per identity
    training_ids =  [f"{int(id)+500:04d}" for id in training_ids]
    for cam in os.listdir(synthetic_dir):
        cam_name = cam[-1]
        for id in os.listdir(os.path.join(synthetic_dir, cam)):
            if id not in training_ids:
                continue
            real_id = str(int(id)+20000)
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
    
    market1501_training_dir = '/data4/toan/projects/DiVE/dataset/Market-1501-v15.09.15/bounding_box_train/'
    training_ids = []
    for image in os.listdir(market1501_training_dir):
        if image[:4] not in training_ids:
            training_ids.append(image[:4])
    
    
    
    move_synthetic_data(args.input, args.output, 3, training_ids)


if __name__ == "__main__":
    main()