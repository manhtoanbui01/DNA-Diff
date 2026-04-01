# import os

# input_dir = "/data4/toan/projects/DiVE/dataset/NightReID_Market1501_ori_stage1_envtoken_stage2_ipadapter_stage3_intensity_full_id/cam7/env301/"

# for identity in os.listdir(input_dir):
#     num_images = len([image for image in os.listdir(os.path.join(input_dir, identity))])
#     if num_images < 3:
#         print(identity)


messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "/data4/toan/projects/DiVE/dataset/NightReID/cam1/0021/0002.jpg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

print(messages[0]['content'][0]["image"])
print(messages[0]['content'][1]["text"])