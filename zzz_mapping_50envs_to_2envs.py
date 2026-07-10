from datasets import load_from_disk

dataset = load_from_disk("/data4/toan/projects/DiVE/dataset/NightReID_Market1501_ori_stage1_envtoken_stage2_ipadapter_stage3_intensity_full_id/")

def remap_environment(example):

    env_str = example["environment"]   # e.g. "env37"

    env_id = int(env_str.replace("env", ""))

    # env1-25 -> env1
    # env26-50 -> env2
    # etc.
    new_env_id = (env_id - 1) // 25 + 1

    example["environment"] = f"env{new_env_id}"

    return example

dataset = dataset.map(remap_environment)
print(sorted(set(dataset["environment"])))

# save
dataset.save_to_disk(
    "/data4/toan/projects/DiVE/dataset/NightReID_Market1501_ori_stage1_envtoken_stage2_ipadapter_stage3_intensity_full_id/hf_dataset_2envs_each_cam"
)