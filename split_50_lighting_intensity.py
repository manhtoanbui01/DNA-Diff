import os
import cv2
import shutil
import math

input_folder = "/data4/toan/datasets/NightReID_all/NightReID_text_inversion/cam/cam6"

cam_name = int(os.path.basename(input_folder).replace("cam", ""))
output_root = f"/data4/toan/datasets/NightReID_all/NightReID_text_inversion/cam_lighting_intensity/cam{cam_name}"

NUM_GROUPS = 50
os.makedirs(output_root, exist_ok=True)

def lighting_score(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return hsv[:, :, 2].mean()

# Collect image scores
items = []
for name in os.listdir(input_folder):
    path = os.path.join(input_folder, name)
    if not os.path.isfile(path):
        continue
    score = lighting_score(path)
    if score is not None:
        items.append((path, name, score))

# Sort dark → bright
items.sort(key=lambda x: x[2])

# Split into groups
chunk_size = math.ceil(len(items) / NUM_GROUPS)

for local_id in range(1, NUM_GROUPS + 1):
    global_id = NUM_GROUPS * (cam_name - 1) + local_id
    
    folder_name = f"{global_id:04d}"
    group_dir = os.path.join(output_root, folder_name)
    os.makedirs(group_dir, exist_ok=True)

    start = (local_id - 1) * chunk_size
    end = start + chunk_size

    if start >= len(items):
        break

    for path, name, score in items[start:end]:
        dst = os.path.join(group_dir, name)
        shutil.copy(path, dst)

print("Done. Images split into lighting groups.")