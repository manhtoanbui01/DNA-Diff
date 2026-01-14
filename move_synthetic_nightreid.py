import os
import shutil
import argparse
from pathlib import Path
import sys
import random
import json



def move_synthetic_data(synthetic_dir, target_dir, id_map, ipc: int = 3):  # ipc: numbers of images per camera per identity
    mapping_cam_names = {'1': 'L1', '2':'L2', '3': 'L3', '4': 'R1', '5': 'R2', '6': 'R3'}
    for cam in os.listdir(synthetic_dir):
        cam_name = mapping_cam_names[cam[-1]]
        for id in os.listdir(os.path.join(synthetic_dir, cam)):
            real_id = id_map[int(id)]
            images = [image for image in os.listdir(os.path.join(synthetic_dir, cam, id))]
            random_ipc_images = random.sample(images, 3)
            for image in random_ipc_images:
                new_name = str(f"{int(real_id):04d}") + cam_name + 'G' + image[1:]
                shutil.copy(os.path.join(synthetic_dir, cam, id, image), os.path.join(target_dir, new_name))
        


def main():
    parser = argparse.ArgumentParser(description='Move synthetic data to target directory')
    
    # Use more explicit argument names with flags
    parser.add_argument('--input', '-i', required=True,
                        help='Path to the input synthetic data directory')
    parser.add_argument('--output', '-o', required=True,
                        help='Path to the output target directory')
    parser.add_argument('--id_mapping_json', required=True, 
                        help="Path to json file containing the id mapping")
    
    args = parser.parse_args()
    
    with open(args.id_mapping_json, 'r') as f:
        data = json.load(f)
        
    id_map = data["mappings"]
    id_map_generated_to_real = {v:k for k,v in id_map.items()}  # the json file contain id mapping from real to generated 
    
    move_synthetic_data(args.input, args.output, id_map_generated_to_real, 3)


if __name__ == "__main__":
    main()