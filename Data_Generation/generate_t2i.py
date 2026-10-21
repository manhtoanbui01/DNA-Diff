
from typing import Any
import os
import sys
import re
import torch
import argparse
import random
import time
import numpy as np
import pandas as pd
import kornia
import torchvision.transforms as T

from semantic_aug.datasets.person_revised import PersonHugDataset
from semantic_aug.augmentations.dreabooth_lora_mt import DreamboothLoraGeneration

from tqdm import tqdm
from PIL import Image
from itertools import product
from collections import defaultdict
from multiprocessing import Process, Queue
from queue import Empty
import sys
# The vendored deep_person_reid/torchreid/__init__.py does `from torchreid
# import ...` (an absolute import), so deep_person_reid/ must itself be on
# sys.path for `torchreid` to resolve to the vendored copy. Both paths are
# derived from this file's location so the repo works from any checkout and
# without installing torchreid into the environment.
_DG_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (_DG_DIR, os.path.join(_DG_DIR, "deep_person_reid")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from deep_person_reid.torchreid.utils import FeatureExtractor
osnet = FeatureExtractor(
    model_name='osnet_ain_x1_0',
    model_path='deep_person_reid/osnet_ain_ms_d_m.pth.tar',
    # device='cuda:0'
)
osnet.model.eval()
osnet.model.requires_grad_(False)

# Reproduce the EXACT OSNet preprocessing used in stage-3 training
# (train_t2i_ipadapter.py). The torchreid FeatureExtractor wrapper resizes
# straight to (256,128) with bilinear interpolation, but training resizes the
# cond image to (512,256) bilinear (via the dataset transform) and then to
# (256,128) with kornia bicubic. Aligning inference to training keeps the
# trained image_proj_model's input distribution consistent.
_OSNET_RESIZE = T.Compose([
    T.Resize((512, 256), interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
    T.Normalize([0.5], [0.5]),   # -> [-1, 1], matching the dataset's self.transform
])


def _osnet_preprocess(x):
    # Identical to preprocess() in train_t2i_ipadapter.py.
    dtype = x.dtype
    x = kornia.geometry.resize(
        x.to(torch.float32), (256, 128),
        interpolation='bicubic', align_corners=True, antialias=False,
    ).to(dtype=dtype)
    x = (x + 1.) / 2.  # [-1,1] -> [0,1]
    x = x.clamp(0, 1)
    x = kornia.enhance.normalize(
        x,
        torch.Tensor([0.485, 0.456, 0.406]),
        torch.Tensor([0.229, 0.224, 0.225]),
    )
    return x


@torch.no_grad()
def extract_osnet_embeds(image_paths):
    """OSNet features for a list of image paths, preprocessed exactly as in training."""
    device = next(osnet.model.parameters()).device
    imgs = torch.stack(
        [_OSNET_RESIZE(Image.open(p).convert('RGB')) for p in image_paths], dim=0
    )
    imgs = _osnet_preprocess(imgs).to(device)  # preprocess on CPU, then move (mirrors training)
    return osnet.model(imgs)


data_dir = os.environ.get("HUG_LOCAL_IMAGE_TRAIN_DIR")

    
def generate_translations(args, in_queue, out_queue, gpu_id):
    model = DreamboothLoraGeneration(
        checkpoint_path = os.path.join(data_dir, "finetune_model", "finetune_stage3_ipadapter", "checkpoint-30000"),
        model_path=args.model_path,
        embed_path=os.path.join(data_dir, "finetune_model", "finetune_stage1_ila", "learned_embeds-steps-100000.bin"),
        prompt=args.prompt,
        guidance_scale=args.guidance_scale,
        mask=args.mask,
        inverted=args.inverted,
        device=f'cuda:{gpu_id}'
    )
    batch_size = args.batch_size

    while True:
        i_lists = []
        cam_names = []
        descriptions = []
        num_cam_tokens = []
        env_ids = []
        class_ids = []
        cond_embeds = []
        strengths = []

        # 1) Pull from queue in mini-batches
        for _ in range(batch_size):
            try:
                # Now the queue items are (i_list, cam_name, desc, class_id, strength)
                q_i_list, q_cam_name, q_desc, q_num_cam_token, q_env_id, q_class_id, q_cond_embed, q_strength = in_queue.get(timeout=1)
                i_lists.append(q_i_list)
                cam_names.append(q_cam_name)
                descriptions.append(q_desc)
                num_cam_tokens.append(q_num_cam_token)
                env_ids.append(q_env_id)
                class_ids.append(q_class_id)
                cond_embeds.append(q_cond_embed)
                strengths.append(q_strength)
            except Empty:
                print("Queue empty, exiting.")
                break

        if len(i_lists) == 0:
            break

        # 2) Generate for each item in the batch
        results = []
        for i_list, cam, desc, num_cam_token, env_id, nm, cond_embed, stg in zip(i_lists, cam_names, descriptions, num_cam_tokens, env_ids, class_ids, cond_embeds, strengths):
            metadata = {
                "camera_name": cam,
                "description": desc,
                "num_cam_token": num_cam_token,
                "env_id": env_id,
                "class_id": nm,
                "cond_embed": cond_embed,
            }

            # We create a random seed for each image in i_list.
            seeds_for_batch = [
                random.randint(0, 2**32 - 1) for _ in i_list
            ]

            # Generate len(i_list) images in one call
            out_images = model(
                metadata=metadata,
                strength=stg,
                resolution=args.resolution,
                num_images_per_prompt=len(i_list),  # <--- We'll add param support below
                seeds=seeds_for_batch,  # <--- pass the random seeds
            )

            # Save results for later
            results.append( (i_list, cam, desc, env_id, nm, stg, out_images) )

        # 3) Save images
        image_name = 0
        for i_list, cam_name, desc, env_id, nm, stg, image_list in results:
            # 'image_list' has len(i_list) images
            for single_i, pil_image in zip(i_list, image_list):
                
                sub_subfolder = f"{int(nm):04d}"
                subfolder = os.path.join(args.output_path, cam_name, sub_subfolder)
                
                os.makedirs(subfolder, exist_ok=True)

            for local_idx, pil_image in enumerate(image_list, start=1):
                filename = f"{env_id}_{image_name}.jpg"
                save_path = os.path.join(subfolder, filename)
                pil_image.save(save_path)
                print(f"Saved {save_path}")
                image_name+=1



def print_instance_attributes(obj):
    """Debug helper."""
    attrs = vars(obj)
    for k, v in attrs.items():
        print(f"{k} = {type(v)} -> {v if len(str(v))<500 else '...'}")
 
def build_queue_via_metadata(
    dataset_obj,
    cond_daytime_path,
    cameras,
    repeat_each_id,
    threshold,
    in_queue,
    images_per_call=1,  # <--- new param
):
    """
    Build a task queue that:
      1) Iterates cameras from metadata.
      2) Iterates class IDs from dataset_obj.label2class.
      3) Only enqueues classes whose numeric ID > 'threshold'.
      4) Repeats each camera->class combination 'repeat_each_id' times.
      5) Now stores BOTH the camera name and the camera description.

    :param dataset_obj: PersonHugDataset or similar,
                       with .camera_to_description, .label2class, .class2label, etc.
    :param cameras: List of camera names to process (e.g. ['cam1', 'cam2']).
                    If None or empty, uses dataset_obj.camera_to_description.keys().
    :param repeat_each_id: How many times to replicate each camera->class pair.
    :param threshold: Numeric threshold for filtering class IDs (e.g. only > threshold).
    :param in_queue: Queue for multiprocessing tasks.
    """
    camera_to_description = dataset_obj.camera_to_description  # e.g. {"cam1": "ABCD1234", ...}                 # e.g. {0: "class_0", 1: "class_1", ...}
    class2label = dataset_obj.class2label 
    # If user doesn't specify cameras, gather them from camera_to_description
    if not cameras:
        cameras = sorted(camera_to_description.keys())

    index_counter = 0
    strength = 1.0

    # Sort class IDs so you process them in ascending order
    all_class_ids = class2label.keys()
    
    camera_num_tokens = dataset_obj.camera_num_tokens
    if args.use_pretrained_weights:
        camera_num_tokens = {'cam1': 3, 'cam2': 2, 'cam3': 4, 'cam4': 2, 'cam5': 3, 'cam6': 3}
        
    label2embed = {}
    for identity in os.listdir(cond_daytime_path):
        label = class2label[str(int(identity))]
        possible_cond_image_paths = [os.path.join(cond_daytime_path, identity, image) for image in os.listdir(os.path.join(cond_daytime_path, identity))]
        possible_cond_embeds = extract_osnet_embeds(possible_cond_image_paths)
        label2embed[label] = possible_cond_embeds
        

    for cam_name in cameras:
        if cam_name not in camera_to_description:
            print(f"[build_queue_via_metadata] Warning: {cam_name} not in camera_to_description!")
            continue

        cam_desc_str = camera_to_description[cam_name]
        num_cam_token = camera_num_tokens[cam_name]
        possible_envs = [f'env{(int(cam_name[-1])-1) * args.num_env_tokens_per_camera + i}' for i in range(1, args.num_env_tokens_per_camera + 1)]

        for class_id in all_class_ids:
            if int(class_id) <= threshold:
                continue
            label = class2label[class_id]
            embeds = label2embed[label]
            

            start_i = index_counter
            end_i = index_counter + repeat_each_id  # e.g. if repeat_each_id=12, range= [0..12)

            # Chunk them
            for chunk_start in range(start_i, end_i, images_per_call):
                chunk_end = min(chunk_start + images_per_call, end_i)
                i_list = list(range(chunk_start, chunk_end))

                # Enqueue i_list (rather than a single i)
                idx = torch.randint(0, embeds.shape[0], (1,))
                cond_embed = embeds[idx].squeeze(0)
                env_id = random.choice(possible_envs)  # choose random environment
                in_queue.put((i_list, cam_name, cam_desc_str, num_cam_token, env_id, class_id, cond_embed, strength))

            # Advance overall index_counter by 'repeat_each_id'
            index_counter += repeat_each_id

    print(f"[build_queue_via_metadata] total tasks queued: {index_counter}")
    
    
def main(args):
    """
    An example end-to-end pipeline:
      1. Prepares output dir, seeds RNG.
      2. Loads dataset.
      3. Builds a queue of tasks sorted by camera description.
      4. Spawns worker processes to generate images.
      5. Tracks progress visually via tqdm.
      6. Generates a final CSV (meta.csv) that indexes generated images.
    """

    # 1) Prepare environment
    os.makedirs(args.output_path, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # 2) GPU IDs and queues for multiprocessing
    gpu_ids = args.gpu_ids
    in_queue = Queue()
    out_queue = Queue()

    # 3) Load your dataset
    #    For example, if "sysu" => PersonHugDataset or similar
    #    The user-provided class handles huggingface dataset creation
    train_dataset = PersonHugDataset(split="train", 
                        seed=args.seed, 
                        examples_per_class=args.examples_per_class, 
                        resolution=args.resolution)

    # 4) Build queue by camera/description
    #    Instead of random sampling by labels, we want camera+id-based tasks.
    # Example threshold=1000, repeat_each_id=3, etc.
    build_queue_via_metadata(
        dataset_obj=train_dataset,
        cond_daytime_path=args.cond_daytime_path,
        cameras=args.cameras,         # e.g. ['cam1','cam2'] or None
        threshold=args.threshold,     # e.g. 1000
        repeat_each_id=args.repeat_each_id,
        in_queue=in_queue,
        images_per_call=args.images_per_call
    )

    # 5) Debug info: dataset stats
    print_instance_attributes(train_dataset)
    num_classes = len(train_dataset.class_names)
    print(f"[main] Dataset has {len(train_dataset)} items and {num_classes} classes.")

    total_tasks = in_queue.qsize()
    print('[main] Number of total tasks:', total_tasks)

    # 6) Debug vs. Normal mode
    if args.debug:
        print("debug run")
        # Single-process run
        generate_translations(args, in_queue, out_queue, gpu_id=gpu_ids[0])

    else:
        print("multi-process run")
        processes = []
        with tqdm(total=total_tasks, desc="Processing") as pbar:
            for process_id, gpu_id_value in enumerate(gpu_ids):
                p = Process(
                    target=generate_translations,
                    args=(args, in_queue, out_queue, gpu_id_value)
                )
                p.start()
                processes.append(p)

            # Update progress bar while processes are alive
            while any(p.is_alive() for p in processes):
                current_queue_size = in_queue.qsize()
                processed = total_tasks - current_queue_size
                pbar.n = processed
                pbar.refresh()
                time.sleep(1)

            for p in processes:
                p.join()
                
    print("All processes finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Inference script")
    parser.add_argument("--output_path", type=str, default="outputs", help="Root path for output data.")
    parser.add_argument("--cond_daytime_path", type=str, default="", help="Root path for output data.")
    parser.add_argument("--num_env_tokens_per_camera", type=int, default=10, help="Root path for output data.")
    parser.add_argument("--confusion_matrix_path", type=str, default=None, help="Path to a precomputed confusion matrix.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--examples_per_class", type=int, default=-1, help="Number of examples per class (if applicable).")
    parser.add_argument("--resolution", type=int, default=512, help="Image resolution.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for generation.")
    parser.add_argument("--prompt", type=str, default="a {conditions} photo of a {name}", help="Base prompt string.")
    parser.add_argument("--mask", type=int, default=0, choices=[0, 1], help="Whether to apply a mask (0=no, 1=yes).")
    parser.add_argument("--inverted", type=int, default=0, choices=[0, 1], help="Whether to invert the mask (0=no, 1=yes).")
    parser.add_argument("--gpu_ids", type=int, nargs="+", default=[0], help="List of GPU IDs to use.")
    parser.add_argument("--debug", action="store_true", help="Use debugging mode (single-process).")
    parser.add_argument("--cameras", nargs="+", default=None, help="List of camera names or descriptions to process (e.g. cam1 cam2). If not set, we gather all unique cameras from the dataset.")
    parser.add_argument("--repeat_each_id", type=int, default=1, help="How many times to enqueue each ID in every camera.")
    parser.add_argument("--threshold", type=int, default=500, help="Ignore classes with numeric ID <= this threshold.")
    parser.add_argument("--images_per_call", type=int, default=1, help="Number of images generated per pipeline call.")
    parser.add_argument("--model_path", type=str, default="CompVis/stable-diffusion-v1-4", help="Base model path.")
    parser.add_argument("--guidance_scale",type=float,default=7.5,help="Guidance scale for classifier-free guidance (how strictly to follow the text prompt, higher = more strict)",)
    parser.add_argument("--use_pretrained_weights", type=lambda x: x.lower() == "true", default=False, help="Whether to use pretrained weights")
    args = parser.parse_args()

    # Please check the corresponding ckpt path before sampling !!! 

    torch.multiprocessing.set_start_method('spawn')
    os.makedirs(args.output_path, exist_ok=True)
    
    main(args)