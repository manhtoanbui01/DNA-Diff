# from safetensors.torch import load_file

# weight_path = "/data4/toan/projects/DiVE/dataset/NightReID_Market1501/finetune_model/finetune_stage2_ipadapter/checkpoint-25000/model.safetensors"

# state_dict = load_file(weight_path)

# print("Number of keys:", len(state_dict))

# for key in state_dict.keys():
#     print(key)


import torch

path = "/data4/toan/projects/DiVE/dataset/NightReID_Market1501_ori_stage1_envtoken_stage2_ipadapter_stage3_intensity_full_id/finetune_model/finetune_stage2_envtoken/learned_embeds-steps-last.bin"

data = torch.load(path, map_location="cpu")["env2placeholder"]

print("Type:", type(data))

if isinstance(data, dict):
    print("Number of keys:", len(data))
    print("Keys:")
    for i, (k,v) in enumerate(data.items()):
        # print(k)
        print(f'{k}:{v}')
        print(type(k))
        if i == 10:
            break
    # print([key for key in data.keys()])

# from datasets import load_from_disk

# dataset = load_from_disk("/data4/toan/projects/DiVE/dataset/NightReID_Market1501_original_model")

# # print(dataset)
# print(dataset[0:10])



# from datasets import load_from_disk

# dataset = load_from_disk("/data4/toan/projects/DiVE/dataset/NightReID_Market1501_ori_stage1_envtoken_stage2_ipadapter_stage3_intensity_full_id/")

# # 1. Access the 'label' column's feature definitions
# label_feature = dataset.features["label"]

# # 2. Print the internal string-to-integer dictionary
# # Note: it's stored in a hidden attribute called _str2int
# print("--- String to Integer Mapping ---")
# mapping = label_feature._str2int
# print(mapping)

# # 3. To see just a few examples (since you have 2001 IDs)
# for i, (name, idx) in enumerate(mapping.items()):
#     print(f"Person ID: {name} maps to Index: {idx}")
#     if i >= 10: break # Stop after 5 examples