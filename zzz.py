import os

input_dir = "/data4/toan/projects/DiVE/dataset/NightReID_Market1501_ori_stage1_envtoken_stage2_ipadapter_stage3_intensity_full_id/cam7/env301/"

for identity in os.listdir(input_dir):
    num_images = len([image for image in os.listdir(os.path.join(input_dir, identity))])
    if num_images < 3:
        print(identity)