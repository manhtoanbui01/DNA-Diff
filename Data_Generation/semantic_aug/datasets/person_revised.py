import os
import random
import torch
from semantic_aug.few_shot_dataset import HugFewShotDataset
from semantic_aug.datasets.utils import IMAGENET_TEMPLATES
from typing import Any, Tuple, Dict

import numpy as np
import pandas as pd
import torchvision.transforms as transforms

from PIL import Image
from shutil import copyfile
from PIL import Image, ImageDraw
from collections import defaultdict
from datasets import load_dataset, load_from_disk

SUPER_CLASS_NAME = 'person'
HUG_LOCAL_IMAGE_TEST_DIR = "/root" #not used, doesn't matter

HUG_LOCAL_IMAGE_TRAIN_DIR = os.getenv("HUG_LOCAL_IMAGE_TRAIN_DIR")
# HUG_LOCAL_IMAGE_TRAIN_DIR = '/path/to/train/data'
if HUG_LOCAL_IMAGE_TRAIN_DIR is None:
    raise ValueError("HUG_LOCAL_IMAGE_TRAIN_DIR env variable is not set.")



def onehot(size, target):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.
    return vec


class PersonHugDataset(HugFewShotDataset):
    super_class_name = SUPER_CLASS_NAME

    def __init__(self, *args, 
                 split: str = "train", 
                 seed: int = 0, 
                 image_train_dir: str = HUG_LOCAL_IMAGE_TRAIN_DIR, 
                 image_test_dir: str = HUG_LOCAL_IMAGE_TEST_DIR, 
                 examples_per_class: int = -1, 
                 synthetic_probability: float = 0.5,
                 return_onehot: bool = False,
                 soft_scaler: float = 0.9,
                 synthetic_dir: str = None,
                 image_size: int = 512,
                 crop_size: int = 448,
                 corrupt_prob=0.0,
                 **kwargs):

        # Choose dataset path based on split
        if split == 'train':
            dataset = load_from_disk(image_train_dir)
        else:
            dataset = load_from_disk(image_test_dir)

        # -------------------------------------------
        # (1) Parse Custom Metadata from info.description
        # -------------------------------------------
        import json

        desc_text = dataset.info.description or ""
        lines = desc_text.split("\n", 1)
        if len(lines) > 1 and lines[0].startswith("Custom metadata:"):
            # The second part lines[1] is presumably the JSON
            metadata_json = lines[1]
            try:
                metadata_dict = json.loads(metadata_json)
            except json.JSONDecodeError as e:
                print("[PersonHugDataset] Warning: Could not decode JSON metadata:", e)
                metadata_dict = {}
        else:
            metadata_dict = {}

        # Store it in self so we can access later
        self.metadata_dict = metadata_dict
        # If present, these might map cameras -> descriptions or cameras -> indices
        self.camera_to_description = metadata_dict.get("camera_to_description", {})
        self.description_to_camera = metadata_dict.get("description_to_camera", {})
        self.environment_to_description = metadata_dict.get("environment_to_description", {})
        self.camera_indices = metadata_dict.get("camera_indices", {})
        self.camera_num_tokens = metadata_dict.get("camera_num_tokens", {})

        # -------------------------------------------
        # (2) Standard RNG, etc.
        # -------------------------------------------
        random.seed(seed)
        np.random.seed(seed)

        # -------------------------------------------
        # (3) Optional: Subsample dataset to [examples_per_class]
        # -------------------------------------------
        if examples_per_class is not None and examples_per_class > 0:
            all_labels = dataset['label']
            label_to_indices = defaultdict(list)
            for i, label in enumerate(all_labels):
                label_to_indices[label].append(i)

            _all_indices = []
            for key, items in label_to_indices.items():
                try:
                    sampled_indices = random.sample(items, examples_per_class)
                except ValueError:
                    print(f"{key}: Sample larger than population, fallback to random.choices")
                    sampled_indices = random.choices(items, k=examples_per_class)
                _all_indices.extend(sampled_indices)

            dataset = dataset.select(_all_indices)

        # -------------------------------------------
        # (4) Save dataset in self
        # -------------------------------------------
        self.dataset = dataset
        self.class_names = [
            name.replace('/', ' ')
            for name in dataset.features['label'].names
        ]
        self.num_classes = len(self.class_names)

        # -------------------------------------------
        # (5) Optional: Corrupt some labels
        # -------------------------------------------
        if corrupt_prob > 0:
            print(f"[PersonHugDataset] corrupt_prob: {corrupt_prob}")
            self.corrupt_labels(corrupt_prob)

        # Create label <-> class dict
        class2label = self.dataset.features['label']._str2int
        self.class2label = {k.replace('/', ' '): v for k, v in class2label.items()}
        self.label2class = {v: k.replace('/', ' ') for k, v in class2label.items()}

        # Keep track of indices for each numeric label
        self.label_to_indices = defaultdict(list)
        for i, label in enumerate(self.dataset['label']):
            self.label_to_indices[label].append(i)

        # -------------------------------------------
        # (6) Call Parent Constructor
        # -------------------------------------------
        super().__init__(
            *args,
            split=split,
            examples_per_class=examples_per_class,
            synthetic_probability=synthetic_probability,
            return_onehot=return_onehot,
            soft_scaler=soft_scaler,
            synthetic_dir=synthetic_dir,
            image_size=image_size,
            crop_size=crop_size,
            **kwargs
        )

    # ---------------------------------------------------------------
    #  (7) Corrupt labels if requested
    # ---------------------------------------------------------------
    def corrupt_labels(self, corrupt_prob):
        labels = np.array(self.dataset['label'])
        np.random.seed(12345)
        mask = np.random.rand(len(labels)) <= corrupt_prob
        rnd_labels = np.random.choice(len(self.class_names), mask.sum())
        labels[mask] = rnd_labels
        labels = [int(x) for x in labels]
        self.dataset = (
            self.dataset
                .remove_columns("label")
                .add_column("label", labels)
                .cast(self.dataset.features)
        )

    # ---------------------------------------------------------------
    #  (8) Basic dataset protocol
    # ---------------------------------------------------------------
    def __len__(self):
        return len(self.dataset)

    def get_image_by_idx(self, idx: int) -> Image.Image:
        return self.dataset[idx]['image'].convert('RGB')

    def get_label_by_idx(self, idx: int) -> int:
        return self.dataset[idx]['label']
    
    def get_cond_embed_by_idx(self, idx: int) -> list:
        return torch.tensor(self.dataset[idx]['cond_embed'])

    def get_metadata_by_idx(self, idx: int) -> dict:
        return dict(
            name=self.label2class[self.get_label_by_idx(idx)],
            super_class=self.super_class_name
        )

class PersonHugDatasetForT2IStep1(torch.utils.data.Dataset):
    """Training dataset for stage 1 (train_t2i_ila.py).

    __getitem__ yields `pixel_values` and `caption` only.

    The prompt's domain slot is the camera pseudo-word taken from the dataset's
    `description` field, and identity placeholders come from whatever the
    trainer assigns to `self.name2placeholder` - stage 1 is what learns them.
    `class_prompts_ratio` defaults to 0.5,
    so roughly half the samples use a raw description line as the prompt.

    See PersonHugDatasetForT2I for the stage 2/3 variant.
    """

    super_class_name = SUPER_CLASS_NAME

    def __init__(self, *args, split: str = "train",
                 seed: int = 0, 
                 image_train_dir: str = HUG_LOCAL_IMAGE_TRAIN_DIR, 
                 max_train_samples: int = -1,
                 class_prompts_ratio: float = 0.5,
                 resolution: Tuple[int, int] = (512, 256),#int = 512,
                 center_crop: bool = False,
                 random_flip: bool = False,
                 examples_per_class: int = -1,
                 **kwargs):

        super().__init__()    

        print(f"image_train_dir: {image_train_dir}")
        dataset = load_from_disk(image_train_dir)

        random.seed(seed)
        np.random.seed(seed)
        if max_train_samples is not None and max_train_samples > 0:
            dataset = dataset.shuffle(seed=seed).select(range(max_train_samples))
        if examples_per_class is not None and examples_per_class > 0:
            all_labels = dataset['label']
            label_to_indices = defaultdict(list)
            for i, label in enumerate(all_labels):
                label_to_indices[label].append(i)

            _all_indices = []
            for key, items in label_to_indices.items():
                try:
                    sampled_indices = random.sample(items, examples_per_class)
                except ValueError:
                    print(f"{key}: Sample larger than population or is negative, use random.choices instead")
                    sampled_indices = random.choices(items, k=examples_per_class)
                    
                label_to_indices[key] = sampled_indices 
                _all_indices.extend(sampled_indices)
            dataset = dataset.select(_all_indices)
        self.class_names = [name.replace('/',' ') for name in dataset.features['label'].names]
        self.dataset = dataset
        self.class2label = self.dataset.features['label']._str2int
        self.label2class = {v: k for k, v in self.class2label.items()} 
        self.num_classes = len(self.class_names)
        self.class_prompts_ratio = class_prompts_ratio
        self.name2placeholder = None
        self.placeholder2name = None
        self.label_to_indices = defaultdict(list)
        for i, label in enumerate(self.dataset['label']):
            self.label_to_indices[label].append(i)
            
        self.transform = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution),
            transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        import json

        desc_text = dataset.info.description or ""
        lines = desc_text.split("\n", 1)
        if len(lines) > 1 and lines[0].startswith("Custom metadata:"):
            # The second part lines[1] is presumably the JSON
            metadata_json = lines[1]
            try:
                metadata_dict = json.loads(metadata_json)
            except json.JSONDecodeError as e:
                print("[PersonHugDataset] Warning: Could not decode JSON metadata:", e)
                metadata_dict = {}
        else:
            metadata_dict = {}

        # Store it in self so we can access later
        self.metadata_dict = metadata_dict
        # If present, these might map cameras -> descriptions or cameras -> indices
        self.camera_to_description = metadata_dict.get("camera_to_description", {})
        self.description_to_camera = metadata_dict.get("description_to_camera", {})
        self.camera_indices = metadata_dict.get("camera_indices", {})


    def __len__(self):
        
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image = self.get_image_by_idx(idx)
        prompt = self.get_prompt_by_idx(idx)
        
        return dict(pixel_values=self.transform(image), caption = prompt)
    
    def get_image_by_idx(self, idx: int) -> Image.Image:

        return self.dataset[idx]['image'].convert('RGB')

    def get_label_by_idx(self, idx: int) -> int:

        return self.dataset[idx]['label']

    def get_prompt_by_idx(self, idx: int) -> str:
        """
        Return a string prompt for the given index,
        using IMAGENET_TEMPLATES
        which has two placeholders: 
        - First for description 
        - Second for name
        """

        # (A) Decide if we build a prompt from label or from random lines in description
        #     Right now, we always do the "class prompt" path if random.random() < self.class_prompts_ratio,
        #     else we pick a random line from the description. 
        #     (Change that ratio or logic as desired.)
        flag = random.random() < self.class_prompts_ratio

        # (B) Access the textual description from the dataset
        domain_description = self.dataset[idx]['description'].strip()
        
        if flag:
            # (C) Optionally get the placeholder name vs. direct class label
            if self.name2placeholder is not None:
                content = self.name2placeholder[self.label2class[self.dataset[idx]['label']]]
            else:
                content = self.label2class[self.dataset[idx]['label']]

            # Append the "super_class_name" if you want, e.g. "person"
            content += " person"

            # (D) Choose a template. Now each template has form "a {} photo of a {}", etc.
            template = random.choice(IMAGENET_TEMPLATES)

            # (E) Fill in two placeholders:
            #     1) the domain description
            #     2) the subject content
            prompt = template.format(domain_description, content)
        else:
            # If not using the "class prompt" approach, pick a random line from description
            # e.g. in case the user's 'description' has multiple lines
            lines = domain_description.split('\n')
            prompt = random.choice(lines) if lines else domain_description

        return prompt

    def get_metadata_by_idx(self, idx: int) -> dict:
        return dict(name=self.label2class[self.get_label_by_idx(idx)])
    

class PersonHugDatasetForT2I(torch.utils.data.Dataset):
    """Training dataset for stage 2 (train_t2i_envtoken.py) and stage 3
    (train_t2i_ipadapter.py).

    Differs from PersonHugDatasetForT2IStep1 in four ways:

    - __getitem__ also yields `cond_pixel_values`, the cross-camera reference
      crop chosen at random from the row's `cond_image` list. Both stages
      collate it into the batch, but only stage 3 consumes it, as the OSNet
      input for the IP-Adapter; in stage 2 it is carried and ignored.
    - get_prompt_by_idx fills the prompt's domain slot with `<env_{N}_{i}>`
      lighting tokens instead of the camera pseudo-word from `description`.
    - get_prompt_by_idx reloads name2placeholder/placeholder2name from stage 1's
      learned_embeds file on every call, overwriting whatever the trainer
      assigned. That path hardcodes stage 1's step count, so it must be kept
      in sync with STEP1_MAX_STEPS (see README section 5.4).
    - `class_prompts_ratio` defaults to 1.0 rather than 0.5.
    """

    super_class_name = SUPER_CLASS_NAME

    def __init__(self, *args, split: str = "train",
                 seed: int = 0, 
                 image_train_dir: str = HUG_LOCAL_IMAGE_TRAIN_DIR, 
                 num_env_tokens_per_camera: int=10,
                 max_train_samples: int = -1,
                 class_prompts_ratio: float = 1.0, # finetuning stage 2-3, always use the full image caption
                 resolution: Tuple[int, int] = (512, 256), #int = 512,
                 center_crop: bool = False,
                 random_flip: bool = False,
                 examples_per_class: int = -1,
                 **kwargs):

        super().__init__()    

        dataset = load_from_disk(image_train_dir)

        random.seed(seed)
        np.random.seed(seed)
        if max_train_samples is not None and max_train_samples > 0:
            dataset = dataset.shuffle(seed=seed).select(range(max_train_samples))
        if examples_per_class is not None and examples_per_class > 0:
            all_labels = dataset['label']
            label_to_indices = defaultdict(list)
            for i, label in enumerate(all_labels):
                label_to_indices[label].append(i)

            _all_indices = []
            for key, items in label_to_indices.items():
                try:
                    sampled_indices = random.sample(items, examples_per_class)
                except ValueError:
                    print(f"{key}: Sample larger than population or is negative, use random.choices instead")
                    sampled_indices = random.choices(items, k=examples_per_class)
                    
                label_to_indices[key] = sampled_indices 
                _all_indices.extend(sampled_indices)
            dataset = dataset.select(_all_indices)
        self.class_names = [name.replace('/',' ') for name in dataset.features['label'].names]
        self.dataset = dataset
        self.class2label = self.dataset.features['label']._str2int
        self.label2class = {v: k for k, v in self.class2label.items()}
        self.num_classes = len(self.class_names)
        self.num_env_tokens_per_camera = num_env_tokens_per_camera
        self.class_prompts_ratio = class_prompts_ratio
        self.name2placeholder = None
        self.placeholder2name = None

        self.label_to_indices = defaultdict(list)
        for i, label in enumerate(self.dataset['label']):
            self.label_to_indices[label].append(i)
            
        self.transform = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution),
            transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        import json

        desc_text = dataset.info.description or ""
        lines = desc_text.split("\n", 1)
        if len(lines) > 1 and lines[0].startswith("Custom metadata:"):
            # The second part lines[1] is presumably the JSON
            metadata_json = lines[1]
            try:
                metadata_dict = json.loads(metadata_json)
            except json.JSONDecodeError as e:
                print("[PersonHugDataset] Warning: Could not decode JSON metadata:", e)
                metadata_dict = {}
        else:
            metadata_dict = {}

        # Store it in self so we can access later
        self.metadata_dict = metadata_dict
        # If present, these might map cameras -> descriptions or cameras -> indices
        self.camera_to_description = metadata_dict.get("camera_to_description", {})
        self.description_to_camera = metadata_dict.get("description_to_camera", {})
        self.camera_indices = metadata_dict.get("camera_indices", {})
        self.camera_num_tokens = metadata_dict.get("camera_num_tokens", {})

        self.env_names = sorted(
            list(set(dataset['environment'])), 
            key=lambda x: int(x.replace('env', ''))
        )  # [env1, env2, ..., env61]
        
        self.environment_names = []
        for env_name in self.env_names:
            cam = f'cam{(int(env_name[3:])-1) // self.num_env_tokens_per_camera + 1}'
            num_tokens = self.camera_num_tokens[cam]
            for i in range(1, num_tokens+1):
                self.environment_names.append(f'{env_name}_{i}')
        
        self.num_envs = len(self.environment_names)
        
        self.env2placeholder = None
        self.placeholder2env = None        

    def __len__(self):
        
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        image = self.get_image_by_idx(idx)
        prompt = self.get_prompt_by_idx(idx)
        cond_image = self.get_cond_image_by_idx(idx)
        

        return dict(pixel_values=self.transform(image), cond_pixel_values=self.transform(cond_image), caption = prompt)

    def get_cond_image_by_idx(self, idx: int) -> list:
        cond_images = self.dataset[idx]["cond_image"]
        cond_image_path = random.choice(cond_images)
        with Image.open(cond_image_path) as img:
        
            return img.convert("RGB")
    
    def get_image_by_idx(self, idx: int) -> Image.Image:

        return self.dataset[idx]['image'].convert('RGB')

    def get_label_by_idx(self, idx: int) -> int:

        return self.dataset[idx]['label']

    def get_prompt_by_idx(self, idx: int) -> str:
        """
        Return a string prompt for the given index,
        using IMAGENET_TEMPLATES
        which has two placeholders: 
        - First for description 
        - Second for name
        """

        # (A) Decide if we build a prompt from label or from random lines in description
        #     class_prompts_ratio defaults to 1.0 here, so the class-prompt path is
        #     effectively always taken. Both branches build the same env-token
        #     domain string; the else branch just omits the identity and template.
        flag = random.random() < self.class_prompts_ratio

        path = f"{HUG_LOCAL_IMAGE_TRAIN_DIR.rstrip('/')}/finetune_model/finetune_stage1_ila/learned_embeds-steps-100000.bin"
        self.name2placeholder = torch.load(path, map_location="cpu")["name2placeholder"]
        self.placeholder2name = torch.load(path, map_location="cpu")["placeholder2name"]
        if flag:
            # (C) Optionally get the placeholder name vs. direct class label
            content = self.name2placeholder[self.label2class[self.dataset[idx]['label']]]

            # Append the "super_class_name" if you want, e.g. "person"
            content += " person"
            
            environment = self.dataset[idx]['environment']
            cam = f'cam{(int(environment[3:])-1) // self.num_env_tokens_per_camera + 1}'
            num_tokens = self.camera_num_tokens[cam]
            domain_description = ""
            for i in range(1, num_tokens+1):
                if i == num_tokens:
                    env_token = f"<env_{environment[3:]}_{i}>"
                else:
                    env_token = f"<env_{environment[3:]}_{i}> "
                domain_description += env_token
        

            # (D) Choose a template. Now each template has form "a {} photo of a {}", etc.
            template = random.choice(IMAGENET_TEMPLATES)

            prompt = template.format(domain_description, content)
            
        else:
            environment = self.dataset[idx]['environment']
            cam = f'cam{(int(environment[3:])-1) // self.num_env_tokens_per_camera + 1}'
            num_tokens = self.camera_num_tokens[cam]
            domain_description = ""
            for i in range(1, num_tokens+1):
                if i == num_tokens:
                    env_token = f"<env_{environment[3:]}_{i}>"
                else:
                    env_token = f"<env_{environment[3:]}_{i}> "
                domain_description += env_token
            prompt = domain_description

        return prompt


    def get_metadata_by_idx(self, idx: int) -> dict:
        return dict(name=self.label2class[self.get_label_by_idx(idx)])

    
if __name__== '__main__':
    PLACEHOLDER_ENV_TEMPLATE = '<env_{}_{}>'
    # HUG_LOCAL_IMAGE_TRAIN_DIR = '/path/to/train/data/'
    train_dataset = PersonHugDatasetForT2I(class_prompts_ratio=0.5)
    environment_names = train_dataset.environment_names
    placeholder_env_tokens = []
    env2placeholder = {}
    for environment_name in environment_names:
        env = int(environment_name[3:-2])
        token_order = int(environment_name[-1])
        placeholder_env_token = PLACEHOLDER_ENV_TEMPLATE.format(env, token_order)
        placeholder_env_tokens.append(placeholder_env_token)
        env2placeholder[environment_name] = placeholder_env_token
    placeholder2env = {placeholder: env for env, placeholder in env2placeholder.items()}
    train_dataset.env2placeholder = env2placeholder
    train_dataset.placeholder2env = placeholder2env
    

    
    for i in range(37000):
        if i < 36000:
            continue
        sample = train_dataset[i]
        print(f"Prompt {i}: {sample['caption']}")
    