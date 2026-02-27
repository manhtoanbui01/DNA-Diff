# from safetensors.torch import load_file

# weight_path = "/data4/toan/projects/DiVE/dataset/NightReID_Market1501/finetune_model/finetune_stage2_ipadapter/checkpoint-25000/model.safetensors"

# state_dict = load_file(weight_path)

# print("Number of keys:", len(state_dict))

# for key in state_dict.keys():
#     print(key)


# import torch

# path = "/data4/toan/projects/DiVE/dataset/NightReID_Market1501/finetune_model/lora_ti/learned_embeds-steps-25000.bin"

# data = torch.load(path, map_location="cpu")["name2placeholder"]

# print("Type:", type(data))

# if isinstance(data, dict):
#     print("Number of keys:", len(data))
#     print("Keys:")
#     for i, (k,v) in enumerate(data.items()):
#         # print(k)
#         print(f'{k}:{v}')
#         if i == 10:
#             break

# from datasets import load_from_disk

# dataset = load_from_disk("/data4/toan/projects/DiVE/dataset/NightReID_Market1501_original_model")

# # print(dataset)
# print(dataset[0:10])



class LoRAIPAdapterAttnProcessor(nn.Module):
    r"""
    Processor for implementing the LoRA attention mechanism using PyTorch 2.0's memory-efficient scaled dot-product
    attention.

    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`, *optional*):
            The number of channels in the `encoder_hidden_states`.
        rank (`int`, defaults to 4):
            The dimension of the LoRA update matrices.
        network_alpha (`int`, *optional*):
            Equivalent to `alpha` but it's usage is specific to Kohya (A1111) style LoRAs.
        kwargs (`dict`):
            Additional keyword arguments to pass to the `LoRALinearLayer` layers.
    """

    def __init__(
        self,
        hidden_size: int,
        cross_attention_dim: Optional[int] = None,
        lora_rank: int = 4,
        lora_alpha: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        rank = lora_rank
        self.rank = rank

        
        self.to_q_lora = LoRALinearLayer(hidden_size, hidden_size, lora_rank, lora_alpha)
        self.to_k_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, lora_rank, lora_alpha)
        self.to_v_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, lora_rank, lora_alpha)
        self.to_out_lora = LoRALinearLayer(hidden_size, hidden_size, lora_rank, lora_alpha)

    def __call__(self, attn: Attention, hidden_states: torch.FloatTensor, *args, **kwargs) -> torch.FloatTensor:
        self_cls_name = self.__class__.__name__

        

        # attn._modules.pop("processor")
        # attn.processor = AttnProcessor2_0()
        # return attn.processor(attn, hidden_states, *args, **kwargs)
        residual = hidden_states

        args = ()

        # if attn.spatial_norm is not None:
        #     hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        
        encoder_hidden_states = kwargs["encoder_hidden_states"]
        if encoder_hidden_states is not None:
            if isinstance(encoder_hidden_states, tuple):
                encoder_hidden_states, ip_hidden_states = encoder_hidden_states
                # print(f'encoder hidden states shape: {encoder_hidden_states.shape}')
                # print(f'ip hidden state shape: {ip_hidden_states.shape}')
            else:
                end_pos = encoder_hidden_states.shape[1] - self.num_tokens[0]
                encoder_hidden_states, ip_hidden_states = (
                    encoder_hidden_states[:, :end_pos, :],
                    encoder_hidden_states[:, end_pos:, :],
                )

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        
        attn.to_q.lora_layer = self.to_q_lora.to(hidden_states.device)
        attn.to_k.lora_layer = self.to_k_lora.to(hidden_states.device)
        attn.to_v.lora_layer = self.to_v_lora.to(hidden_states.device)
        attn.to_out[0].lora_layer = self.to_out_lora.to(hidden_states.device)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

  
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states