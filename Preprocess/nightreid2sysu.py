import os
import shutil
import argparse
import json

def extract_environment_name(image_name, ref_folder="/data4/toan/datasets/NightReID_all/NightReID_text_inversion/cam_lighting_merge_outlier/"):
    for camera_name in os.listdir(ref_folder):
        camera_path = os.path.join(ref_folder, camera_name)
        for environment in os.listdir(camera_path):
            environment_image_names = [name for name in os.listdir(os.path.join(camera_path, environment))]
            if image_name in environment_image_names:
                return (int(camera_name[-1])-1) * 50 + int(environment) + 1  # environment: [1,2,3,...,301]
    raise ValueError(f"the isn't {image_name} in {ref_folder}")


def main():
    parser = argparse.ArgumentParser(description="Convert NightReID dataset to SYSU-MM01 format")

    # Hyperparameters:
    parser.add_argument(
        "--nightreid_training_dir",
        type=str,
        default="/data4/toan/datasets/NightReID_all/NightReID/bounding_box_train",
        help="Path of the training dir of NightReID dataset "
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dataset/NightReID_Market1501_ori_stage1_envtoken_stage2_ipadapter_stage3",
        help="Output path of the NightReID dataset"
    )
    args = parser.parse_args()
    
    
    nightreid_training_dir = args.nightreid_training_dir
    output_dir = args.output_dir
    
    id_to_images = {}  # key: id, value: list of image filenames
    
    all_files = sorted(os.listdir(nightreid_training_dir))
    
    for image_file in all_files:
        if image_file[:4].isdigit():
            nightreid_id = int(image_file[:4])
        else:
            print(f"Skipping file that does not meet the format: {image_file}")
            continue
        
        if nightreid_id not in id_to_images:
            id_to_images[nightreid_id] = []
        
        id_to_images[nightreid_id].append(image_file)
    
    id_mapping = {}
    for i, nightreid_id in enumerate(id_to_images.keys(), start=1):
        new_id = i
        id_mapping[nightreid_id] = new_id
        
        image_list = id_to_images[nightreid_id]
        image_list.sort()
        for idx, image_file in enumerate(image_list, start=1):
            old_image_path = os.path.join(nightreid_training_dir, image_file)
            camera_id = int(image_file[5]) + 3 if image_file[4] == 'R' else int(image_file[5])
            environment_name = extract_environment_name(image_file)
            new_image_path = os.path.join(output_dir, f'cam{camera_id}', f'env{environment_name}',f"{new_id:04d}", f"{idx:04d}.jpg")
            os.makedirs(os.path.dirname(new_image_path), exist_ok=True)
            shutil.copy(old_image_path, new_image_path)
    
    # 5. Update train_id.txt
    train_id_file = os.path.join(output_dir, "exp", "train_id.txt")
    os.makedirs(os.path.dirname(train_id_file), exist_ok=True)
    # Create the file if it does not exist
    if not os.path.exists(train_id_file):
        with open(train_id_file, "w") as f:
            f.write("")  # or initial content
    
    new_train_ids = [str(i) for i in range(1, len(id_to_images.keys()) + 1)]
    
    with open(train_id_file, "w") as f:
        f.write(",".join(new_train_ids))
        
    # 6. Save the mapping of old and new IDs to a JSON file with annotations
    mapping_save_dir = os.path.join(output_dir, "exp")
    mapping_file = os.path.join(mapping_save_dir, "id_mapping.json")
    # Prepare the JSON data structure
    mapping_data = {
        "title": "ID Mapping from NightReID to SYSU-MM01 form",
        "annotations": {
            "original_id": "NightReID ID",
            "new_id": "SYSU-MM01 ID"
        },
        "mappings": {str(old_id): new_id for old_id, new_id in id_mapping.items()},
        "total_mappings": len(id_mapping)
    }
    # Write the JSON file with proper formatting
    with open(mapping_file, "w") as f:
        json.dump(mapping_data, f, indent=4)
    
    
    print(f"The mapping of old and new IDs has been saved to {mapping_file}.")


if __name__ == "__main__":
    main()
    
        