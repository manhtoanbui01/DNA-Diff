# from safetensors.torch import load_file

# weight_path = "/data4/toan/projects/DiVE/dataset/NightReID_Market1501/finetune_model/finetune_stage2_ipadapter/checkpoint-25000/model.safetensors"

# state_dict = load_file(weight_path)

# print("Number of keys:", len(state_dict))

# for key in state_dict.keys():
#     print(key)


# import torch

# path = "/data4/toan/projects/DiVE/dataset/NightReID_Market1501_ori_stage1_envtoken_stage2_ipadapter_stage3_intensity_full_id/finetune_model/finetune_stage2_envtoken/learned_embeds-steps-last.bin"

# data = torch.load(path, map_location="cpu")["learned_embeds_dict"]

# print("Type:", type(data))

# if isinstance(data, dict):
#     print("Number of keys:", len(data))
#     print("Keys:")
#     for i, (k,v) in enumerate(data.items()):
#         # print(k)
#         # print(f'{k}:{v}')
#         # print(type(k))
#         v_dtype = getattr(v, "dtype", type(v))
#         print(f'{i} | Key: {k} | Value Dtype: {v_dtype}')
#         if i == 10:
#             break
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



# compare 2 lora_ipadapter checkpoint file

from safetensors.torch import load_file
import torch

ckpt1_path = "/data4/toan/projects/DiVE/dataset/NightReID_Market1501_ori_stage1_envtoken_stage2_ipadapter_stage3_intensity_full_id/finetune_model/finetune_stage3_ipadapter_real/checkpoint-25000/model.safetensors"
ckpt2_path = "/data4/toan/projects/DiVE/dataset/NightReID_Market1501_ori_stage1_envtoken_stage2_ipadapter_stage3_intensity_full_id/finetune_model/finetune_stage3_ipadapter/checkpoint-15000/model.safetensors"

ckpt1 = load_file(ckpt1_path)
ckpt2 = load_file(ckpt2_path)

keys1 = set(ckpt1.keys())
keys2 = set(ckpt2.keys())

print("Only in ckpt1:", keys1 - keys2)
print("Only in ckpt2:", keys2 - keys1)
print("Common keys:", len(keys1 & keys2))

# diffs = {}

# for k in ckpt1.keys():
#     if k in ckpt2:
#         diff = torch.mean(torch.abs(ckpt1[k] - ckpt2[k])).item()
#         diffs[k] = diff

# # Sort by largest difference
# sorted_diffs = sorted(diffs.items(), key=lambda x: x[1], reverse=True)

# for k, v in sorted_diffs[:]:
#     print(f"{k}: {v}")

stats = {}

for k in ckpt1.keys():
    if k in ckpt2:
        w1 = ckpt1[k]
        w2 = ckpt2[k]

        mean1 = w1.mean().item()
        mean2 = w2.mean().item()
        mean_diff = abs(mean1 - mean2)

        abs_diff = torch.mean(torch.abs(w1 - w2)).item()

        stats[k] = {
            "mean1": mean1,
            "mean2": mean2,
            "mean_diff": mean_diff,
            "abs_diff": abs_diff,
        }

# Sort by mean difference
sorted_stats = sorted(stats.items(), key=lambda x: x[1]["mean_diff"], reverse=True)

for k, v in sorted_stats:
    print(f"{k}: mean1={v['mean1']:.6f}, mean2={v['mean2']:.6f}, "
          f"mean_diff={v['mean_diff']:.6f}, abs_diff={v['abs_diff']:.6f}")
    
    
# Compare 2 learned_embed_dict files:

# import torch

# path1 = "/data4/toan/projects/DiVE/dataset/NightReID_Market1501_ori_stage1_envtoken_stage2_ipadapter_stage3_intensity_full_id/finetune_model/finetune_stage1_dive/learned_embeds-steps-5000.bin"
# path2 = "/data4/toan/projects/DiVE/dataset/NightReID_Market1501_ori_stage1_envtoken_stage2_ipadapter_stage3_intensity_full_id/finetune_model/finetune_stage1_dive/learned_embeds-steps-15000.bin"

# embeds1 = torch.load(path1, map_location="cpu")["learned_embeds_dict"]
# embeds2 = torch.load(path2, map_location="cpu")["learned_embeds_dict"]

# keys1 = set(embeds1.keys())
# keys2 = set(embeds2.keys())

# print("Only in file1:", keys1 - keys2)
# print("Only in file2:", keys2 - keys1)
# print("Common tokens:", len(keys1 & keys2))

# diffs = {}

# for k in embeds1:
#     if k in embeds2:
#         diff = torch.mean(torch.abs(embeds1[k] - embeds2[k])).item()
#         diffs[k] = diff

# # Top changed tokens
# for k, v in sorted(diffs.items(), key=lambda x: x[1], reverse=True)[:100]:
#     print(f"{k}: {v}")