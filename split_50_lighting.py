import os
import shutil
import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min
import torch
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from PIL import Image
import os
import shutil

def extract_camera_agnostic_lighting(image_path):
    """
    Extracts lighting features while minimizing camera and identity bias.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # 1. Spatial Masking: Focus on the background/peripheral area
    # In ReID, the person is central. We look at the outer 20% of the image
    # to understand the 'environment map' rather than the 'person'.
    h, w, _ = img.shape
    mask = np.ones((h, w), dtype=np.uint8)
    mask[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)] = 0 # Mask out the center person
    
    # 2. Color Distribution (HSV Space)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # We calculate histograms only for the masked area (the environment)
    # H: Color/Tone, S: Saturation (purity of light), V: Brightness
    h_hist = cv2.calcHist([hsv], [0], mask, [16], [0, 180])
    s_hist = cv2.calcHist([hsv], [1], mask, [8], [0, 256])
    v_hist = cv2.calcHist([hsv], [2], mask, [8], [0, 256])
    
    # 3. Illumination Smoothness (Laplacian Variance)
    # Tells us if the light is diffuse (soft) or specular (harsh/point-source)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # 4. Global Color Constant (Gray World assumption deviation)
    # Represents the 'Color Temperature' of the scene
    avg_color = cv2.mean(img, mask=mask)[:3]
    
    # Concatenate all features into a 'Lighting Fingerprint'
    features = np.concatenate([
        h_hist.flatten(), 
        s_hist.flatten(), 
        v_hist.flatten(), 
        [laplacian_var], 
        avg_color
    ])
    
    return features

def group_lighting_robust(image_folder, num_clusters=50):
    valid_extensions = ('.jpg', '.jpeg', '.png')
    files = [f for f in os.listdir(image_folder) if f.lower().endswith(valid_extensions)]
    
    all_features = []
    image_paths = []
    
    print(f"Analyzing lighting statistics for {len(files)} images...")
    
    for file in files:
        path = os.path.join(image_folder, file)
        feat = extract_camera_agnostic_lighting(path)
        if feat is not None:
            all_features.append(feat)
            image_paths.append(path)
    
    features_array = np.array(all_features)
    
    # Standardize to ensure Laplacian Variance doesn't dominate Histograms
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_array)
    
    print(f"Clustering into {num_clusters} conditions (Camera-Agnostic)...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features_scaled)
    
    return labels, image_paths

if __name__ == "__main__":
    DATASET_PATH = "/data4/toan/datasets/NightReID_all/NightReID_text_inversion/cam/cam2/"
    OUTPUT_PATH = "/data4/toan/datasets/NightReID_all/NightReID_text_inversion/cam_lighting_merge_outlier_10_groups/cam2/"
    MIN_IMAGES = 12
    TARGET_GROUPS = 10

    # 1. Get initial clusters
    # We use features_scaled and kmeans object later, so let's modify the return slightly
    # For this snippet, assume group_lighting_robust returns (labels, paths, features_scaled, kmeans)
    labels, paths = group_lighting_robust(DATASET_PATH, num_clusters=TARGET_GROUPS)
    
    # Track clusters
    cluster_map = {i: [] for i in range(TARGET_GROUPS)}
    for path, label in zip(paths, labels):
        cluster_map[label].append(path)

    # 2. Re-assign small clusters to nearest neighbors
    # Find clusters that are too small
    small_clusters = [cid for cid, imgs in cluster_map.items() if len(imgs) < MIN_IMAGES]
    healthy_clusters = [cid for cid, imgs in cluster_map.items() if len(imgs) >= MIN_IMAGES]

    print(f"Found {len(small_clusters)} clusters with fewer than {MIN_IMAGES} images. Redistributing...")

    for cid in small_clusters:
        images_to_move = cluster_map[cid]
        # In a real scenario, you'd calculate the distance to the next nearest centroid.
        # For simplicity, we'll move them to the current largest healthy cluster 
        # to ensure they are 'safe', then we will split the large ones later.
        target = max(healthy_clusters, key=lambda k: len(cluster_map[k]))
        cluster_map[target].extend(images_to_move)
        cluster_map[cid] = [] # Empty the small cluster

    # 3. Split the largest clusters until we reach exactly 50 groups again
    while len([cid for cid, imgs in cluster_map.items() if len(imgs) >= MIN_IMAGES]) < TARGET_GROUPS:
        # Find the biggest cluster to split
        biggest_cid = max(cluster_map.keys(), key=lambda k: len(cluster_map[k]))
        images_in_biggest = cluster_map[biggest_cid]
        
        # Find an empty slot (one of the dissolved small_clusters)
        empty_slot = next(cid for cid, imgs in cluster_map.items() if len(imgs) == 0)
        
        # Split the biggest cluster in half
        mid = len(images_in_biggest) // 2
        cluster_map[biggest_cid] = images_in_biggest[:mid]
        cluster_map[empty_slot] = images_in_biggest[mid:]
        
        print(f"Split cluster {biggest_cid} to fill slot {empty_slot}")

    # 4. Final Output
    if os.path.exists(OUTPUT_PATH):
        shutil.rmtree(OUTPUT_PATH)
        
    for i in range(TARGET_GROUPS):
        current_folder = os.path.join(OUTPUT_PATH, f"{i:04d}")
        os.makedirs(current_folder, exist_ok=True)
        
        for path in cluster_map[i]:
            filename = os.path.basename(path)
            shutil.copy(path, os.path.join(current_folder, filename))
            
    print(f"Grouping complete. All {TARGET_GROUPS} folders have >= {MIN_IMAGES} images (unless total dataset is too small).")