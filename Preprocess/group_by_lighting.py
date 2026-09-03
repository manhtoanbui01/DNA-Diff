"""
Split a nighttime ReID training set into environment groups by lighting level.

Each camera's images are sorted by brightness (mean HSV V channel) and divided
into NUM_GROUPS equal-sized groups. The result is written as:

    <OUTPUT_PATH>/cam{c}/env{e}/{new_id:04d}/<original filename>

where e = (c - 1) * NUM_GROUPS + level, and level runs from 1 (darkest)
to NUM_GROUPS (brightest).

The relabelled identities are also written to <OUTPUT_PATH>/exp/train_id.txt as
a single comma-separated line, the format the Preprocess scripts read.
"""

import os
import shutil

import cv2
import numpy as np

INPUT_PATH = "/data4/toan/datasets/NightReID_all/NightReID/z_NightReID/bounding_box_train/"
OUTPUT_PATH = "/data4/toan/projects/DNA-Diff/dataset/NightReID_Market1501"

NUM_GROUPS = 10  # number of environment tokens per camera


def lighting_score(image_path):
    """Mean of the HSV value channel. Returns None if the file is not an image."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 2].mean())


def parse_name(filename):
    """
    Read the identity and camera from a filename.

    L1, L2, L3 -> cam 1, 2, 3
    R1, R2, R3 -> cam 4, 5, 6

    Example: '0042R3_...' -> identity 42, cam 6
    """
    identity = int(filename[:4])
    side = filename[4]
    if side not in ("L", "R"):
        raise ValueError(f"unexpected camera prefix in {filename}")
    cam_id = int(filename[5])
    if side == "R":
        cam_id += 3
    return identity, cam_id


# --------------------------------------------------
# 0. Start from a clean output folder
# --------------------------------------------------
# Without this, folders from a previous run with a different NUM_GROUPS
# would still be present and would be mixed into the new dataset.

if os.path.exists(OUTPUT_PATH):
    shutil.rmtree(OUTPUT_PATH)
os.makedirs(OUTPUT_PATH)


# --------------------------------------------------
# 1. Read every image once and group them by camera
# --------------------------------------------------

images_by_camera = {}   # cam_id -> list of (filename, identity, score)

for filename in sorted(os.listdir(INPUT_PATH)):

    score = lighting_score(os.path.join(INPUT_PATH, filename))
    if score is None:
        continue        # not an image, e.g. .DS_Store

    identity, cam_id = parse_name(filename)

    if cam_id not in images_by_camera:
        images_by_camera[cam_id] = []
    images_by_camera[cam_id].append((filename, identity, score))

total = sum(len(v) for v in images_by_camera.values())
print(f"read {total} images from {len(images_by_camera)} cameras")


# --------------------------------------------------
# 2. Divide each camera into NUM_GROUPS lighting levels
# --------------------------------------------------

level_of = {}   # filename -> lighting level (1 = darkest)

for cam_id in sorted(images_by_camera):

    images = images_by_camera[cam_id]
    images.sort(key=lambda item: item[2])   # darkest -> brightest

    # Equal-frequency split: every group gets the same number of images,
    # so a larger NUM_GROUPS means fewer images per environment token.
    positions = np.array_split(np.arange(len(images)), NUM_GROUPS)

    for level, group in enumerate(positions, start=1):
        for i in group:
            filename = images[i][0]
            level_of[filename] = level

    print(f"  cam{cam_id}: {len(images)} images, "
          f"V from {images[0][2]:.1f} to {images[-1][2]:.1f}")


# --------------------------------------------------
# 3. Relabel identities to a contiguous range 1..M
# --------------------------------------------------

all_identities = sorted({identity
                         for images in images_by_camera.values()
                         for _, identity, _ in images})

identity_mapping = {old: new
                    for new, old in enumerate(all_identities, start=1)}

print(f"relabelled {len(identity_mapping)} identities")


# --------------------------------------------------
# 4. Write the new identities to exp/train_id.txt
# --------------------------------------------------
# One comma-separated line of plain integers, which is what the Preprocess
# scripts expect (they read it with .strip().split(',')).

train_id_path = os.path.join(OUTPUT_PATH, "exp", "train_id.txt")
os.makedirs(os.path.dirname(train_id_path), exist_ok=True)

new_ids = sorted(identity_mapping.values())

with open(train_id_path, "w") as f:
    f.write(",".join(str(new_id) for new_id in new_ids))

print(f"Wrote {len(new_ids)} ids to {train_id_path}")


# --------------------------------------------------
# 5. Copy each image into its camera/environment/identity folder
# --------------------------------------------------

for cam_id in sorted(images_by_camera):
    for filename, identity, _ in images_by_camera[cam_id]:

        env = (cam_id - 1) * NUM_GROUPS + level_of[filename]
        new_id = identity_mapping[identity]

        output_dir = os.path.join(OUTPUT_PATH,
                                  f"cam{cam_id}",
                                  f"env{env}",
                                  f"{new_id:04d}")
        os.makedirs(output_dir, exist_ok=True)

        shutil.copy(os.path.join(INPUT_PATH, filename),
                    os.path.join(output_dir, filename))

print(f"Done, images written to {OUTPUT_PATH}")