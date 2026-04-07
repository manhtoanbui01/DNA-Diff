# import os

# input_dir = "/data4/toan/projects/DiVE/dataset/NightReID_Market1501_ori_stage1_envtoken_stage2_ipadapter_stage3_intensity_full_id/cam7/env301/"

# for identity in os.listdir(input_dir):
#     num_images = len([image for image in os.listdir(os.path.join(input_dir, identity))])
#     if num_images < 3:
#         print(identity)

PLACEHOLDER_TOKEN_TEMPLATE = '<class_{}>'
PLACEHOLDER_ENV_TEMPLATE = '<env_{}>'
INITIALIZER_TOKEN = 42170
INITIALIZER_ENV_TOKEN = 42171
HUG_LOCAL_IMAGE_TRAIN_DIR = "/data4/toan/projects/DiVE/dataset/NightReID_Market1501_ori_stage1_envtoken_stage2_ipadapter_stage3_intensity_full_id/"

from transformers import CLIPTextModel, CLIPTokenizer
import torch
import os

tokenizer = CLIPTokenizer.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="tokenizer"
    )
text_encoder = CLIPTextModel.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="text_encoder"
    )

text_encoder.text_model.embeddings.token_embedding.weight.data = text_encoder.get_input_embeddings().weight.data.float()


# print(tokenizer.decode([42171]))
# print(tokenizer.convert_ids_to_tokens(42171))

placeholder_tokens = [PLACEHOLDER_TOKEN_TEMPLATE.format(i) for i in range(2001)] #TEMPLATE '<class_{}>'

num_added_tokens = tokenizer.add_tokens(placeholder_tokens)

# Resize the token embeddings as we are adding new special tokens to the tokenizer
text_encoder.resize_token_embeddings(len(tokenizer)) #增加embedding的长度，[49408, 768]->[50814, 768]

# Initialise the newly added placeholder token with the embeddings of the initializer token
token_embeds = text_encoder.get_input_embeddings().weight.data #[50814, 768]
initializer_token_id = INITIALIZER_TOKEN #42170
placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens) #49408, 49409, 49410, 49411, 49412 ... 496607 #正常的embedding到49407结束
with torch.no_grad(): #placeholder_tokens ['<class_0>', ...'<class_1404>', '<class_1405>']
    for token_id in placeholder_token_ids:
        token_embeds[token_id] = token_embeds[initializer_token_id].clone() #把newly added token 赋予INITIALIZER_TOKEN对应的embedding的值   


data_dir = HUG_LOCAL_IMAGE_TRAIN_DIR
text_encoder_weight_path = os.path.join(data_dir, "finetune_model", "finetune_stage1_dive", "learned_embeds-steps-25000.bin")
learned_embeds = torch.load(text_encoder_weight_path, map_location="cpu")["learned_embeds_dict"]

with torch.no_grad():
    for token, embed in learned_embeds.items():
        token_id = tokenizer.convert_tokens_to_ids(token)
        # print(f"Person token: {token}")
        # print(f"Loaded weight for token id {token_id}")
        token_embeds[token_id] = embed
        # print(f"Person token id: {token_id}")

placeholder_env_tokens = [PLACEHOLDER_ENV_TEMPLATE.format(i) for i in range(301)]  #TEMPLATE '<env_{}>'
# import pdb;pdb.set_trace()
num_added_tokens = tokenizer.add_tokens(placeholder_env_tokens)

# Resize the token embeddings as we are adding new special tokens to the tokenizer

# Initialise the newly added placeholder token with the embeddings of the initializer token
text_encoder.resize_token_embeddings(len(tokenizer)) # [49408, 768]->[50814, 768]
token_embeds = text_encoder.get_input_embeddings().weight.data #[50814, 768]
initializer_env_id = INITIALIZER_ENV_TOKEN #42171
placeholder_env_ids = tokenizer.convert_tokens_to_ids(placeholder_env_tokens) #49408, 49409, 49410, 49411, 49412 ... 496607 #正常的embedding到49407结束
with torch.no_grad(): #placeholder_tokens ['<env_0>', ...'<env_299>', '<env_300>']
    for token_id in placeholder_env_ids:
        token_embeds[token_id] = token_embeds[initializer_env_id].clone() #把newly added token 赋予INITIALIZER_TOKEN对应的embedding的值
        # print(f"Env token id: {token_id}")
        
index_no_updates = torch.ones((len(tokenizer),), dtype=torch.bool) #[49608] # no update tokens
index_no_updates[placeholder_env_ids] = False  # update placeholder_env_ids
# print(f'Placeholder_env_ids: {placeholder_env_ids}')


print(index_no_updates.sum())
print(len(index_no_updates))
