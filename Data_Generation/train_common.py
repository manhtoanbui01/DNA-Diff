"""Helpers shared by the three training stages.

These functions were duplicated verbatim across train_t2i_ila.py,
train_t2i_envtoken.py and train_t2i_ipadapter.py; the bodies here are
byte-identical to the copies they replace.

save_progress() and save_progress_env() differ only in which mapping they
serialise - the identity variant writes name2placeholder/placeholder2name, the
environment variant writes env2placeholder/placeholder2env. Those key names are
read back by the next stage and by generate_t2i.py, so the two must stay
separate rather than being collapsed into one parameterised function.
"""

import torch
import kornia
from accelerate.logging import get_logger

logger = get_logger(__name__, log_level="INFO")


def preprocess(x):
    dtype = x.dtype
    x = kornia.geometry.resize(x.to(torch.float32), (256, 128), interpolation='bicubic', align_corners=True, antialias=False).to(dtype=dtype)   # not bf16
    x = (x + 1.) / 2.  # [-1,1] -> [0,1]
    x = x.clamp(0, 1)
    x = kornia.enhance.normalize(x, torch.Tensor([0.485, 0.456, 0.406]),
                                 torch.Tensor([0.229, 0.224, 0.225]))
    return x

def extract_features(model, image_tensor):
    """
    Extract osnet features from a tensor.

    Args:
        model: osnet model
        image_tensor: torch.Tensor of shape [B, 3, H, W] or [3, H, W] in range [-1,1]
        transform: optional torchvision transform to preprocess image_tensor

    Returns:
        features: torch.Tensor of shape [B, D]
    """

    image_tensor = preprocess(image_tensor)

    # Ensure batch dimension
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)  # [1, 3, H, W]

    image_tensor = image_tensor.cuda()
    # image_tensor = image_tensor.to(dtype=next(model.parameters()).dtype)

    features = model(image_tensor)  # [B, D]

    return features

def save_progress(text_encoder, placeholder_token_ids, accelerator, args, name2placeholder, placeholder2name, save_path):
    logger.info("Saving embeddings")
    learned_embeds = (
        accelerator.unwrap_model(text_encoder)
        .get_input_embeddings()
        .weight[placeholder_token_ids]
    )
    learned_embeds_dict = {}
    for learned_embed, (name, placeholder) in zip(learned_embeds, name2placeholder.items()):
        learned_embeds_dict[placeholder] = learned_embed
    
    torch.save(dict(learned_embeds_dict = learned_embeds_dict, name2placeholder=name2placeholder, placeholder2name=placeholder2name), save_path)

def save_progress_env(text_encoder, placeholder_env_ids, accelerator, args, env2placeholder, placeholder2env, save_path):
    logger.info("Saving embeddings")
    learned_embeds = (
        accelerator.unwrap_model(text_encoder)
        .get_input_embeddings()
        .weight[placeholder_env_ids]
    )
    learned_embeds_dict = {}
    for learned_embed, (env, placeholder) in zip(learned_embeds, env2placeholder.items()):
        learned_embeds_dict[placeholder] = learned_embed
    
    torch.save(dict(learned_embeds_dict = learned_embeds_dict, env2placeholder=env2placeholder, placeholder2env=placeholder2env), save_path)
