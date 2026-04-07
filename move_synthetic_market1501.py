import os
import shutil
import argparse
from pathlib import Path
import sys
import random
import json



def move_synthetic_data(synthetic_dir, target_dir, ipc: int = 3):  # ipc: numbers of images per camera per identity
    for cam in os.listdir(synthetic_dir):
        cam_name = cam[-1]
        # if cam_name != "6":
        #     continue
        for id in os.listdir(os.path.join(synthetic_dir, cam)):
            real_id = str(int(id)+10000)
            images = [image for image in os.listdir(os.path.join(synthetic_dir, cam, id))]
            if len(images) < ipc:
                random_ipc_images = images
            else:
                random_ipc_images = random.sample(images, ipc)
            for image in random_ipc_images:
                new_name = str(f"{int(real_id):05d}") + f"_c{cam_name}" + "_" + image
                shutil.copy(os.path.join(synthetic_dir, cam, id, image), os.path.join(target_dir, new_name))

def move_synthetic_data_cuhksysu(synthetic_dir, target_dir, ipc: int = 3):  # ipc: numbers of images per camera per identity
    id2imgpath = {}
    for cam in os.listdir(synthetic_dir):
        for identity in os.listdir(os.path.join(synthetic_dir, cam)):
            if identity not in id2imgpath:
                id2imgpath[identity] = []
            for image in os.listdir(os.path.join(synthetic_dir, cam, identity)):
                id2imgpath[identity].append(os.path.join(synthetic_dir, cam, identity, image))

    for identity in id2imgpath:
        random_ipc_images = random.sample(id2imgpath[identity], ipc)
        new_id = str(int(identity) + 20000)
        for i, image in enumerate(random_ipc_images):
            old_name = os.path.basename(image)
            new_name = str(f"{int(new_id):05d}") + f"_c7_{i}.png"
            shutil.copy(image, os.path.join(target_dir, new_name))


def main():
    parser = argparse.ArgumentParser(description='Move synthetic data to target directory')
    
    # Use more explicit argument names with flags
    parser.add_argument('--input', '-i', required=True,
                        help='Path to the input synthetic data directory')
    parser.add_argument('--output', '-o', required=True,
                        help='Path to the output target directory')
    
    args = parser.parse_args()
    
    
    move_synthetic_data(args.input, args.output, 3)


if __name__ == "__main__":
    main()