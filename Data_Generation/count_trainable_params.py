"""
Count the number of *trainable* parameters in the DiVE 3-stage pipeline.

What each stage actually optimizes (read directly from the trainers):

  Stage 1  (train_t2i_dive.py)
      - LoRA on every UNet attn processor         (rank 128)      [args.learning_rate]
      - <class_*> textual-inversion token rows                    [args.ti_learning_rate]
        (the whole embedding matrix is handed to the optimizer, but `index_no_updates`
         masks the gradients so ONLY the placeholder rows are actually updated)

  Stage 2  (train_t2i_envtoken.py)
      - LoRA on every UNet attn processor         (rank 128)   <-- SAME layers as stage 1
      - <env_*> textual-inversion token rows

  Stage 3  (train_t2i_ipadapter.py)
      - LoRA is FROZEN (lines 710-714: requires_grad_=False for every "lora" param)
      - IP-Adapter cross-attn projections to_k_ip / to_v_ip   (trainable)
      - image_proj_model (ImageProjModel)                     (trainable)
      - text encoder frozen -> NO textual inversion here

Dedup rule requested ("if the same parameter is trained in stage 1/2/3, count once"):
    The LoRA layers are trained in BOTH stage 1 and stage 2 (identical modules,
    continued fine-tuning). In the union count they are counted ONCE.
    Everything else is disjoint (class-emb only in S1, env-emb only in S2,
    IP-adapter only in S3).

    union(1+2+3) = LoRA  +  class_emb  +  env_emb  +  ip_kv  +  image_proj_model

Run:
    python count_trainable_params.py \
        --model_path runwayml/stable-diffusion-v1-5 \
        --rank 128 \
        --stage1_embeds /path/finetune_stage1_dive/learned_embeds-steps-100000.bin \
        --stage2_embeds /path/finetune_stage2_envtoken/learned_embeds-steps-40000.bin
"""

import argparse
import glob
import os

import torch
from diffusers import UNet2DConditionModel
from diffusers.models.attention_processor import LoRAAttnProcessor

from lora_ipadapter import ImageProjModel, LoRAIPAdapterAttnProcessor


def count_params(module, only_trainable=False):
    """Sum of numel over a module's parameters."""
    return sum(
        p.numel() for p in module.parameters()
        if (p.requires_grad or not only_trainable)
    )


def build_lora_procs(unet, rank):
    """Replicates the stage 1 / stage 2 processor construction (all LoRAAttnProcessor)."""
    procs = {}
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
        procs[name] = LoRAAttnProcessor(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            rank=rank,
        )
    return procs


def build_stage3_procs(unet, rank):
    """Replicates the stage 3 processor construction (LoRAIPAdapter on cross-attn, LoRA on self-attn),
    then freezes every 'lora' param exactly as the trainer does (lines 710-714)."""
    procs = {}
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
            procs[name] = LoRAIPAdapterAttnProcessor(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                lora_rank=rank,
            )
        else:
            procs[name] = LoRAAttnProcessor(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                rank=rank,
            )
    # Freeze LoRA, only finetune IPAdapter (train_t2i_ipadapter.py:710-714)
    for module in procs.values():
        for pname, param in module.named_parameters():
            if "lora" in pname:
                param.requires_grad = False
    return procs


def count_embeds(path_or_glob, hidden_dim, fallback_num_tokens):
    """Count textual-inversion parameters = (#learned token rows) x hidden_dim.

    Prefers reading the saved learned_embeds .bin (exact), else uses --num_*_tokens * hidden_dim.
    Returns (n_params, n_tokens, source_str).
    """
    path = None
    if path_or_glob:
        matches = sorted(glob.glob(path_or_glob))
        if matches:
            path = matches[-1]
    if path and os.path.isfile(path):
        d = torch.load(path, map_location="cpu")["learned_embeds_dict"]
        n_params = sum(v.numel() for v in d.values())
        return n_params, len(d), f"from checkpoint {os.path.basename(path)}"
    if fallback_num_tokens is not None:
        return fallback_num_tokens * hidden_dim, fallback_num_tokens, "from --num_*_tokens x hidden_dim"
    return 0, 0, "NOT FOUND (pass --stageN_embeds or --num_*_tokens)"


def fmt(n):
    return f"{n:,} ({n / 1e6:.3f} M)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="/data4/toan/projects/DiVE/stable-diffusion-v1-5")
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--hidden_dim", type=int, default=768, help="CLIP text embed dim (SD-1.5 = 768).")
    ap.add_argument("--ip_embeddings_dim", type=int, default=512)
    ap.add_argument("--ip_extra_context_tokens", type=int, default=4)
    ap.add_argument("--stage1_embeds", default=None, help="learned_embeds .bin (class tokens). Glob ok.")
    ap.add_argument("--stage2_embeds", default=None, help="learned_embeds .bin (env tokens). Glob ok.")
    ap.add_argument("--num_class_tokens", type=int, default=None, help="fallback if no stage1 .bin")
    ap.add_argument("--num_env_tokens", type=int, default=None, help="fallback if no stage2 .bin")
    args = ap.parse_args()

    print(f"Loading UNet from {args.model_path} (CPU) ...")
    unet = UNet2DConditionModel.from_pretrained(
        args.model_path, subfolder="unet", local_files_only=True
    )

    # ---- LoRA (shared stage 1 & 2) ----
    lora_procs = build_lora_procs(unet, args.rank)
    n_lora = sum(count_params(p) for p in lora_procs.values())

    # ---- Stage 3 IP-adapter + image proj ----
    s3_procs = build_stage3_procs(unet, args.rank)
    n_ip_kv = sum(count_params(p, only_trainable=True) for p in s3_procs.values())
    image_proj = ImageProjModel(
        cross_attention_dim=unet.config.cross_attention_dim,
        ip_embeddings_dim=args.ip_embeddings_dim,
        ip_extra_context_tokens=args.ip_extra_context_tokens,
    )
    n_proj = count_params(image_proj)

    # ---- Textual inversion embeddings ----
    n_class_emb, n_class_tok, src1 = count_embeds(
        args.stage1_embeds, args.hidden_dim, args.num_class_tokens
    )
    n_env_emb, n_env_tok, src2 = count_embeds(
        args.stage2_embeds, args.hidden_dim, args.num_env_tokens
    )

    # ---- Per-stage totals ----
    stage1 = n_lora + n_class_emb
    stage2 = n_lora + n_env_emb
    stage3 = n_ip_kv + n_proj
    union = n_lora + n_class_emb + n_env_emb + n_ip_kv + n_proj  # LoRA counted once

    line = "-" * 68
    print(line)
    print("COMPONENT BREAKDOWN")
    print(line)
    print(f"  LoRA (rank {args.rank}, all attn procs)      : {fmt(n_lora)}")
    print(f"  class-token embeddings ({n_class_tok} tok)   : {fmt(n_class_emb)}  [{src1}]")
    print(f"  env-token embeddings   ({n_env_tok} tok)     : {fmt(n_env_emb)}  [{src2}]")
    print(f"  IP-adapter to_k_ip+to_v_ip (trainable)       : {fmt(n_ip_kv)}")
    print(f"  image_proj_model (ImageProjModel)            : {fmt(n_proj)}")
    print(line)
    print("PER-STAGE TRAINABLE PARAMETERS")
    print(line)
    print(f"  Stage 1 only (DiVE)  = LoRA + class_emb           : {fmt(stage1)}")
    print(f"  Stage 2              = LoRA + env_emb             : {fmt(stage2)}")
    print(f"  Stage 3              = ip_kv + image_proj         : {fmt(stage3)}")
    print(line)
    print("WHOLE PIPELINE (1+2+3), shared LoRA counted ONCE")
    print(line)
    print(f"  union = LoRA + class_emb + env_emb + ip_kv + proj : {fmt(union)}")
    print(line)


if __name__ == "__main__":
    main()
