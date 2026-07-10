from datasets import load_from_disk

dataset = load_from_disk(
    "/data4/toan/projects/DiVE/dataset/NightReID_Market1501_ori_stage1_envtoken_stage2_ipadapter_stage3_intensity_full_id/"
)

def remap_environment(example):
    env_id = int(example["environment"].replace("env", ""))

    # Which block of 50 are we in?
    block = (env_id - 1) // 50

    # Position within the current block: 1~50
    pos = (env_id - 1) % 50 + 1

    if pos <= 30:
        # First 30 envs -> 10 groups of 3
        group_in_block = (pos - 1) // 3 + 1
    else:
        # Last 20 envs -> 5 groups of 4
        group_in_block = (pos - 31) // 4 + 11

    # Global group index
    new_env_id = block * 15 + group_in_block

    example["environment"] = f"env{new_env_id}"
    return example

dataset = dataset.map(remap_environment)

print(sorted(set(dataset["environment"])))

dataset.save_to_disk(
    "/data4/toan/projects/DiVE/dataset/NightReID_Market1501_ori_stage1_envtoken_stage2_ipadapter_stage3_intensity_full_id/hf_dataset_15envs_each_50block"
)