# from safetensors.torch import load_file

# weight_path = "/data4/toan/projects/DiVE/dataset/NightReID_Market1501/finetune_model/lora_ti/checkpoint-10000/model_1.safetensors"

# state_dict = load_file(weight_path)

# print("Number of keys:", len(state_dict))

# for key in state_dict.keys():
#     print(key)


import torch

path = "/data4/toan/projects/DiVE/dataset/NightReID_Market1501/finetune_model/lora_ti/learned_embeds-steps-25000.bin"

data = torch.load(path, map_location="cpu")["name2placeholder"]

print("Type:", type(data))

if isinstance(data, dict):
    print("Number of keys:", len(data))
    print("Keys:")
    for i, (k,v) in enumerate(data.items()):
        # print(k)
        print(f'{k}:{v}')
        if i == 10:
            break