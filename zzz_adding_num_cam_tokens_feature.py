from datasets import load_from_disk
from transformers import CLIPTokenizer

# =========================
# Paths
# =========================
dataset_path = "/data4/toan/projects/DiVE/dataset/Night600_CUHK03_ori_stage1_envtoken_stage2_ipadapter_stage3_intensity/"
save_path = "/data4/toan/projects/DiVE/dataset/Night600_CUHK03_ori_stage1_envtoken_stage2_ipadapter_stage3_intensity/test/"

# =========================
# Load dataset
# =========================
dataset = load_from_disk(dataset_path)

# =========================
# Load tokenizer
# =========================
tokenizer = CLIPTokenizer.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    subfolder="tokenizer"
)

# =========================
# Count tokens
# =========================
num_camera_tokens = []

for sample in dataset:

    text = sample["description"]

    token_ids = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]

    num_camera_tokens.append(len(token_ids))

# =========================
# Add new column
# =========================
dataset = dataset.add_column(
    "num_camera_tokens",
    num_camera_tokens
)

# =========================
# Save dataset
# =========================
dataset.save_to_disk(save_path)

print(dataset)
print("Done!")



# test code

# from datasets import load_from_disk

# # load dataset
# dataset = load_from_disk("/data4/toan/datasets/test_error/test/")

# # create dictionary
# camera_to_num_tokens = {}

# for sample in dataset:

#     camera = sample["camera"]
#     num_tokens = sample["num_camera_tokens"]

#     # keep first occurrence
#     if camera not in camera_to_num_tokens:
#         camera_to_num_tokens[camera] = num_tokens

# # print dictionary
# print(camera_to_num_tokens)