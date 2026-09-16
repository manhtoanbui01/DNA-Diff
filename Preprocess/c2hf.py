import argparse
import os
import random
import string
import json
from PIL import Image
import datasets
from datasets import Dataset, Features, Image as ImageFeature, Sequence
from transformers import CLIPTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a dataset from images and store metadata in dataset.info.description."
    )
    parser.add_argument("--dataset", type=str, required=True, help="Name/identifier of the dataset (e.g. 'nightreid').")
    parser.add_argument("--base_path", type=str, required=True, help="Base path to the input dataset.")
    parser.add_argument("--num_cameras", type=int, required=True, help="Number of cameras in the dataset.")
    return parser.parse_args()

def read_ids(file_path):
    with open(file_path, 'r') as f:
        return f.read().strip().split(',')
    
import random
import string

def generate_unique_descriptions(count):
    """
    Generate 'count' random codes of length 4 with no character overlap among them.
    If count > 15, it will raise a ValueError, because we only have 62 distinct
    alphanumeric characters in total.
    """
    all_chars = list(string.ascii_letters + string.digits)  # 62 unique chars
    random.shuffle(all_chars)  # randomize the pool of characters

    max_codes = len(all_chars) // 4  # each code needs 4 unique chars
    if count > max_codes:
        raise ValueError(
            f"Cannot generate {count} completely different codes of length 4. "
            f"Maximum possible is {max_codes}."
        )

    codes = []
    start = 0
    for _ in range(count):
        # Take 4 characters without overlap
        chunk = all_chars[start : start + 4]
        codes.append("".join(chunk))
        start += 4

    return codes

    
    

def main():
    args = parse_args()
    # Extract arguments
    current_dataset = args.dataset
    base_path = args.base_path
    output_path = args.base_path
    num_cameras = args.num_cameras
    num_night_cameras = num_cameras - 1  # all daytime images are consider captured by the same camera
    
    train_id_path = os.path.join(base_path, "exp", "train_id.txt")
    all_ids = read_ids(train_id_path)
    id_to_label = {f"{int(id_str):04d}": i for i, id_str in enumerate(all_ids)}  # {'0001': 0, '0002': 1, ...}
    
    tokenizer = CLIPTokenizer.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="tokenizer"
    )
    
    
    # Prepare data
    data = {"image": [], "label": [], "environment": [], "cond_image": [], "description": [], "camera": []}
    
    # Generate per-camera code descriptions
    camera_descriptions = generate_unique_descriptions(num_cameras) 
    
    # Create mappings
    camera_to_description = {f'cam{i}': desc for i, desc in enumerate(camera_descriptions, 1)}
    description_to_camera = {desc: f"cam{i}" for i, desc in enumerate(camera_descriptions, 1)}
    
    
    
    
    # Saving these mappings to a file
    camera_mappings = {
        "dataset": current_dataset,
        "num_cameras": num_cameras,
        "camera_to_description": camera_to_description,
        "description_to_camera": description_to_camera,
    }
    mapping_file = f"camera_environment_description_mappings_{current_dataset}.json"
    with open(mapping_file, "w") as f:
        json.dump(camera_mappings, f)
        
    # choose all images of the same identity with different camera for condition
    def find_possible_cond_images(cam_name, cam_path, identity):
        possible_cond_image_paths = []
        other_cam_names = [x for x in [f"cam{i+1}" for i in range(num_night_cameras)] if x != cam_name]
        for cond_cam_name in other_cam_names:
            cond_cam_path = os.path.join(os.path.dirname(cam_path), cond_cam_name)
            for env in os.listdir(cond_cam_path):
                env_path = os.path.join(cond_cam_path, env)
                if identity in os.listdir(env_path):
                    for image in os.listdir(os.path.join(env_path, identity)):
                        possible_cond_image_paths.append(os.path.join(env_path, identity, image))
        return possible_cond_image_paths
            
    
    def process_images(camera_path, camera_name):
        description = camera_to_description[camera_name]
        # token_ids = tokenizer(description, add_special_tokens=False, truncation=False)["input_ids"]
        # num_camera_tokens = len(token_ids)
        for environment in os.listdir(camera_path):
            # environment_description = environment_to_description[environment]
            environment_path = os.path.join(camera_path, environment)
            for identity in os.listdir(environment_path):
                identity_path = os.path.join(environment_path, identity)
                possible_cond_image_paths = find_possible_cond_images(camera_name, camera_path, identity)
                if len(possible_cond_image_paths) == 0:
                    print(f'Person {identity} cam {camera_name} has 0 possible condition image, skip it')
                    continue
                for img_name in os.listdir(identity_path):
                    img_path = os.path.join(identity_path, img_name)
                    try:
                        with Image.open(img_path) as img:
                            width, height = img.size
                            if height >= 2 * width:                        
                                data["image"].append(img.convert("RGB"))
                                data["label"].append(id_to_label[identity])
                                data["cond_image"].append(possible_cond_image_paths)
                                data["description"].append(description)
                                data["camera"].append(camera_name)
                                # data["num_camera_tokens"].append(num_camera_tokens)
                                data["environment"].append(environment)
                            else:
                                print(f"Skipping image {img_path} due to insufficient height")
                    except Exception as e:
                        print(f"Error processing image {img_path}: {e}")
                
                
    def process_market1501_images(camera_path, camera_name):
        description = camera_to_description[camera_name]
        # token_ids = tokenizer(description, add_special_tokens=False, truncation=False)["input_ids"]
        # num_camera_tokens = len(token_ids)
        environment = os.listdir(camera_path)[0]
        environment_path = os.path.join(camera_path, environment)
        for identity in os.listdir(environment_path):
            identity_path = os.path.join(environment_path, identity)
            for image in os.listdir(identity_path):
                img_path = os.path.join(identity_path, image)
                possible_cond_image_paths = [os.path.join(identity_path, img) for img in os.listdir(identity_path) if img != image]
                if len(possible_cond_image_paths) == 0:
                    print(f'identity {identity} image {image}  has 0 possible condition image, skip it')
                    continue
                
                with Image.open(img_path) as img:
                    data["image"].append(img.convert("RGB"))
                    data["label"].append(id_to_label[identity])
                    data["cond_image"].append(possible_cond_image_paths)
                    data["description"].append(description)
                    data["camera"].append(camera_name)
                    # data["num_camera_tokens"].append(num_camera_tokens)
                    data["environment"].append("env61")
            
    # Process images by environment
    camera_paths = [os.path.join(base_path, f"cam{i}") for i in range(1, num_cameras + 1)]
    if os.path.join(base_path, f"cam{num_cameras}") in camera_paths:
        camera_paths = [cam for cam in camera_paths if cam != os.path.join(base_path, f"cam{num_cameras}")]
        process_market1501_images(camera_path=os.path.join(base_path, f"cam{num_cameras}"), camera_name=f"cam{num_cameras}")
        
    for i, camera_path in enumerate(camera_paths, 1):
        camera_name = f"cam{i}"
        process_images(camera_path, camera_name)

    
    # Build the dataset
    dataset = Dataset.from_dict(
        data,
        features=datasets.Features(
            {
                "image": ImageFeature(),
                "label": datasets.ClassLabel(names=all_ids),
                "cond_image": Sequence(datasets.Value("string")),
                "description": datasets.Value("string"),
                "camera": datasets.Value("string"),
                # "num_camera_tokens": datasets.Value("int32"),
                "environment": datasets.Value("string"),
            }
        )
    )
    
    # Create index mappings for quick retrieval by camera
    camera_indices = {}
    for cam_name in camera_to_description.keys():
        camera_indices[cam_name] = [i for i,c in enumerate(data['camera']) if c == cam_name]
    
    camera_num_tokens = {}
    for cam_name, cam_desc in camera_to_description.items():
        token_ids = tokenizer(cam_desc, add_special_tokens=False, truncation=False)["input_ids"]
        num_camera_tokens = len(token_ids)
        camera_num_tokens[cam_name] = num_camera_tokens
    
    
    
    # Prepare metadata dict
    metadata_dict = {
        "dataset": current_dataset,
        "num_cameras": num_cameras,
        "camera_indices": camera_indices,
        "camera_to_description": camera_to_description,
        "description_to_camera": description_to_camera,
        "camera_num_tokens": camera_num_tokens,
    }
    
    # Store metadata as JSON in dataset.info.description
    metadata_json = json.dumps(metadata_dict, indent=2)
    dataset.info.description = f"Custom metadata:\n{metadata_json}"

    # Save the dataset to the combined path
    dataset.save_to_disk(output_path)

    # Print summary
    print(f"Dataset '{current_dataset}' created with {len(dataset)} images.")
    print(f"Number of cameras: {num_cameras}")
    print(f"Unique camera descriptions created: {len(set(data['description']))}")
    print(f"Dataset saved to: {output_path}")
    
    

if __name__ == "__main__":
    main()