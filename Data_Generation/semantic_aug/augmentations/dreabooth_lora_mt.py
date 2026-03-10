import sys
sys.path.append('/data/zhicai/code/da-fusion/')
from semantic_aug.generative_augmentation import GenerativeMixup
from diffusers import StableDiffusionImg2ImgPipeline,DPMSolverMultistepScheduler
from diffusers import StableDiffusionInpaintPipeline
from diffusers import StableDiffusionPipeline
from transformers import (
    CLIPFeatureExtractor, 
    CLIPTextModel, 
    CLIPTokenizer
)
from diffusers.utils import logging
from PIL import Image, ImageOps

from typing import Any, Tuple, Callable, List
from torch import autocast
from scipy.ndimage import maximum_filter

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append("/data4/toan/projects/DiVE/Data_Generation")
from lora_ipadapter import ImageProjModel, LoRAIPAdapterAttnProcessor
from safetensors.torch import load_file
from diffusers.models.attention_processor import LoRAAttnProcessor
os.environ["http_proxy"]="http://localhost:8890"
os.environ["https_proxy"]="http://localhost:8890"
os.environ["WANDB_DISABLED"] = "true"
ERROR_MESSAGE = "Tokenizer already contains the token {token}. \
Please pass a different `token` that is not already in the tokenizer."

def format_name(name):
    return f"<{name.replace(' ', '_')}>"

def load_diffmix_embeddings(embed_path: str,
                            text_encoder: CLIPTextModel,
                            tokenizer: CLIPTokenizer,
                            device="cuda",
            ):

    embedding_ckpt = torch.load(embed_path, map_location='cpu')
    learned_embeds_dict = embedding_ckpt["learned_embeds_dict"]
    name2placeholder = embedding_ckpt["name2placeholder"]
    placeholder2name = embedding_ckpt["placeholder2name"]

    name2placeholder = {k.replace('/',' ').replace('_',' '): v for k, v in name2placeholder.items()}
    placeholder2name = {v: k.replace('/',' ').replace('_',' ') for k, v in name2placeholder.items()} 
    
    for token, token_embedding in learned_embeds_dict.items():

        # add the token in tokenizer
        num_added_tokens = tokenizer.add_tokens(token)
        assert num_added_tokens > 0, ERROR_MESSAGE.format(token=token)
    
        # resize the token embeddings
        text_encoder.resize_token_embeddings(len(tokenizer))
        added_token_id = tokenizer.convert_tokens_to_ids(token)

        # get the old word embeddings
        embeddings = text_encoder.get_input_embeddings()

        # get the id for the token and assign new embeds
        embeddings.weight.data[added_token_id] = \
            token_embedding.to(embeddings.weight.dtype)

    return name2placeholder, placeholder2name

def identity(*args):
    return args

class IdentityMap:
    def __getitem__(self, key):
        return key
    
class DreamboothLoraMixup(GenerativeMixup):

    pipe = None  # global sharing is a hack to avoid OOM

    def __init__(self, lora_path: str, 
                 model_path: str = "runwayml/stable-diffusion-v1-5",
                 embed_path: str = None,
                 prompt: str = "a photo of a {name}",
                 format_name: Callable = format_name,
                 guidance_scale: float = 7.5,
                 mask: bool = False,
                 inverted: bool = False,
                 mask_grow_radius: int = 16,
                 disable_safety_checker: bool = True,
                 revision: str = None,
                 device="cuda", 
                 **kwargs):

        super(DreamboothLoraMixup, self).__init__()

        if DreamboothLoraMixup.pipe is None:

            PipelineClass = (StableDiffusionInpaintPipeline 
                             if mask else 
                             StableDiffusionImg2ImgPipeline)


            DreamboothLoraMixup.pipe = PipelineClass.from_pretrained(
                model_path, use_auth_token=True,
                revision=revision, 
                local_files_only=True,
                torch_dtype=torch.float16
            ).to(device)

            scheduler = DPMSolverMultistepScheduler.from_config(DreamboothLoraMixup.pipe.scheduler.config, local_files_only=True)
            self.placeholder2name = None
            self.name2placeholder = None
            if embed_path is not None:
                self.name2placeholder, self.placeholder2name = load_diffmix_embeddings(embed_path, DreamboothLoraMixup.pipe.text_encoder, DreamboothLoraMixup.pipe.tokenizer)
            if lora_path is not None:
                DreamboothLoraMixup.pipe.load_lora_weights(lora_path)
            DreamboothLoraMixup.pipe.scheduler = scheduler
 
            print(f"successfuly load lora weights from {lora_path}! ! ! ")

            logging.disable_progress_bar()
            self.pipe.set_progress_bar_config(disable=True)

            if disable_safety_checker:
                self.pipe.safety_checker = None

        self.prompt = prompt
        self.guidance_scale = guidance_scale
        self.format_name = format_name

        self.mask = mask
        self.inverted = inverted
        self.mask_grow_radius = mask_grow_radius

        self.erasure_word_name = None

    def forward(self, image: Image.Image, label: int, 
                metadata: dict, strength: float=0.5, resolution=512) -> Tuple[Image.Image, int]:

        canvas = [img.resize((resolution, resolution), Image.BILINEAR) for img in image]
        name = metadata.get("name", "") 

        if self.name2placeholder is not None:
            name = self.name2placeholder[name] 
        if metadata.get("super_class", None) is not None:
            name  = name + ' ' + metadata.get("super_class", "")
        prompt = self.prompt.format(name=name)

        print(prompt)

        if self.mask: assert "mask" in metadata, \
            "mask=True but no mask present in metadata"
        
        # word_name = metadata.get("name", "").replace(" ", "")

        kwargs = dict(
            image=canvas,
            prompt=[prompt], 
            strength=strength, 
            guidance_scale=self.guidance_scale,
            num_inference_steps=25,
            num_images_per_prompt=len(canvas)
        )
        
        if self.mask:  # use focal object mask
            mask_image = metadata["mask"].resize((resolution, resolution), Image.NEAREST)

            mask_image = Image.fromarray(
                maximum_filter(np.array(mask_image), 
                               size=self.mask_grow_radius))

            if self.inverted:

                mask_image = ImageOps.invert(
                    mask_image.convert('L')).convert('1')

            kwargs["mask_image"] = mask_image

        has_nsfw_concept = True
        while has_nsfw_concept:
            with autocast("cuda"):
                outputs = self.pipe(**kwargs)

            has_nsfw_concept = (
                self.pipe.safety_checker is not None 
                and outputs.nsfw_content_detected[0]
            )
        canvas = []
        for orig, out in zip(image, outputs.images):
            canvas.append(out.resize(orig.size, Image.BILINEAR))
        return canvas, label


class DreamboothLoraGeneration(GenerativeMixup):

    pipe = None  # global sharing is a hack to avoid OOM

    def __init__(self, checkpoint_path: str, 
                 model_path: str = "runwayml/stable-diffusion-v1-5",
                 embed_path: str = None,
                 prompt: str = "a photo of a {name}",
                 format_name: Callable = format_name,
                 guidance_scale: float = 7.5,
                 mask: bool = False,
                 inverted: bool = False,
                 mask_grow_radius: int = 16,
                 disable_safety_checker: bool = True,
                 revision: str = None,
                 device="cuda", 
                 **kwargs):

        super(DreamboothLoraGeneration, self).__init__()

        if DreamboothLoraGeneration.pipe is None:

            PipelineClass = StableDiffusionPipeline

            DreamboothLoraGeneration.pipe = PipelineClass.from_pretrained(
                model_path, use_auth_token=True,
                revision=revision, 
                local_files_only=True,
                torch_dtype=torch.float16
            ).to(device)

            scheduler = DPMSolverMultistepScheduler.from_config(DreamboothLoraGeneration.pipe.scheduler.config, local_files_only=True)
            self.placeholder2name = None
            self.name2placeholder = None
            # embed_path = os.path.join(os.path.dirname(os.path.normpath(checkpoint_path)), f'learned_embeds-steps-last.bin')
            embed_path = "/data4/toan/projects/DiVE/dataset/NightReID_Market1501/finetune_model/lora_ti/learned_embeds-steps-last.bin"
            self.name2placeholder, self.placeholder2name = load_diffmix_embeddings(embed_path, DreamboothLoraGeneration.pipe.text_encoder, DreamboothLoraGeneration.pipe.tokenizer)
            
            unet = DreamboothLoraGeneration.pipe.unet
            lora_ipadapter_attn_procs = {}
            for name in unet.attn_processors.keys():
                cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
                if name.startswith("mid_block"):
                    hidden_size = unet.config.block_out_channels[-1]
                elif name.startswith("up_blocks"):
                    block_id = int(name[len("up_blocks.")])
                    hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
                elif name.startswith("down_blocks"):
                    block_id = int(name[len("down_blocks.")])
                    hidden_size = unet.config.block_out_channels[block_id]

                if cross_attention_dim:
                    lora_ipadapter_attn_procs[name] = LoRAIPAdapterAttnProcessor(
                        hidden_size=hidden_size,
                        cross_attention_dim=cross_attention_dim,
                        lora_rank=128,
                    ).to(device)
                else:
                    lora_ipadapter_attn_procs[name] = LoRAAttnProcessor(
                        hidden_size=hidden_size,
                        cross_attention_dim=cross_attention_dim,
                        rank=128,
                    ).to(device)
            unet.set_attn_processor(lora_ipadapter_attn_procs)
            # load lora and lora_ipadapter weights
            lora_ipadapter_state = load_file(os.path.join(checkpoint_path, 'model.safetensors'))
            missing_keys = []
            unexpected_keys = []
            for name, proc in unet.attn_processors.items():
                proc_state = {
                    k.replace(f"{name}.", ""): v
                    for k, v in lora_ipadapter_state.items()
                    if k.startswith(name)
                }

                if len(proc_state) == 0:
                    continue

                m, u = proc.load_state_dict(proc_state, strict=False)
                missing_keys.extend([f"{name}.{k}" for k in m])
                unexpected_keys.extend([f"{name}.{k}" for k in u])

            print("Missing keys lora_ipadapter_state:", missing_keys)
            print("Unexpected keys lora_ipadapter_state:", unexpected_keys)
            
            self.image_proj_model = ImageProjModel(
                cross_attention_dim=unet.config.cross_attention_dim,
                ip_embeddings_dim=512,
                ip_extra_context_tokens=4,
            )
            image_proj_state = load_file(os.path.join(checkpoint_path, 'model_1.safetensors'))
            missing, unexpected = self.image_proj_model.load_state_dict(
                image_proj_state,
                strict=False
            )
            self.image_proj_model.to(next(self.pipe.unet.parameters()).device)
            print("Missing keys image_proj_state:", missing)
            print("Unexpected keys image_proj_state:", unexpected)
            
            DreamboothLoraGeneration.pipe.scheduler = scheduler
 
            print(f"successfuly load lora ipadapter weights from {checkpoint_path}! ! ! ")

            logging.disable_progress_bar()
            self.pipe.set_progress_bar_config(disable=True)

            if disable_safety_checker:
                self.pipe.safety_checker = None

        self.prompt = prompt
        self.guidance_scale = guidance_scale
        self.format_name = format_name

        self.mask = mask
        self.inverted = inverted
        self.mask_grow_radius = mask_grow_radius

        self.erasure_word_name = None

    def forward(
        self,
        metadata: dict,
        strength: float = 0.5,
        resolution: int = 512,
        num_images_per_prompt: int = 1,
        seeds: List[int] = None  # <-- new
    ) -> List[Image.Image]:
        """
        Generate images from text prompts (metadata) and conditional image (to add more detailed).
        If 'seeds' is passed and len(seeds) == num_images_per_prompt,
        each image in the batch uses a distinct random seed.
        """

        name = metadata.get("class_id", "")
        if self.name2placeholder is not None:
            name = self.name2placeholder.get(name, name)

        name += " person"
        desc = metadata.get("description", "")
        env_desc = metadata.get("environment_description", "")
        env_desc += " conditions"
        # Single prompt string
        prompt_str = self.prompt.format(name=name, description=desc, conditions=env_desc)
        print("Using prompt:", prompt_str)
        
        device = next(self.pipe.unet.parameters()).device
        cond_embed = metadata.get("cond_embed", "").to(device)
        ipadapter_embed = self.image_proj_model(cond_embed)
        if num_images_per_prompt > 1:
            ipadapter_embed = ipadapter_embed.repeat_interleave(
                num_images_per_prompt, dim=0
            )

        # Base kwargs
        kwargs = dict(
            ipadapter_embed=ipadapter_embed,
            guidance_scale=self.guidance_scale,
            num_inference_steps=50,
            height=resolution,
            width=resolution // 2,
        )

        if not seeds:
            # No seeds given => single-seed approach (or random seed from pipeline)
            # We rely on num_images_per_prompt=N to generate N latents
            kwargs["prompt"] = [prompt_str]
            kwargs["num_images_per_prompt"] = num_images_per_prompt
            generator = None  # or you can pass one generator if you want reproducibility
        else:
            # If we have seeds, we'll create a separate generator for each image
            if len(seeds) != num_images_per_prompt:
                raise ValueError(f"Expected {num_images_per_prompt} seeds, got {len(seeds)}")
            
            # We'll pass the same prompt repeated N times
            prompt_list = [prompt_str] * num_images_per_prompt
            generator_list = []
            for seed_val in seeds:
                gen = torch.Generator(device=self.pipe.device.type)
                gen.manual_seed(seed_val)
                generator_list.append(gen)
            kwargs["prompt"] = prompt_list
            kwargs["generator"] = generator_list
            # set num_images_per_prompt=1 for each prompt
            kwargs["num_images_per_prompt"] = 1

        # If you have mask logic, handle it here
        if self.mask:
            raise NotImplementedError("Mask logic is disabled in these examples.")

        has_nsfw_concept = True
        outputs = None
        with torch.no_grad():
            while has_nsfw_concept:
                with autocast("cuda"):
                    outputs = self.pipe(**kwargs)
                has_nsfw_concept = (
                    self.pipe.safety_checker is not None
                    and outputs.nsfw_content_detected[0]
                )

        return outputs.images


if __name__ == '__main__':
    ds = DreamboothLoraMixup('/data/zhicai/code/da-fusion/outputs/finetune/sd-cub-model-lora/checkpoint-1000/pytorch_model.bin',
                             'runwayml/stable-diffusion-v1-5',
                             )
    print('result')