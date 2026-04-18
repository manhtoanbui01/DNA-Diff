import torch
import os
from diffusers import StableDiffusionPipeline
from daam import trace

from Data_Generation.semantic_aug.augmentations.dreabooth_lora_mt_stage2 import DreamboothLoraGeneration


MODEL_ID = "runwayml/stable-diffusion-v1-5"
PROMPT = "a <env_0> photo of a person"
OUTPUT_DIR = "./daam_results"
DEVICE = "cuda"

os.makedirs(OUTPUT_DIR, exist_ok=True)