# extract_lidar_min_dist.py
import os
import numpy as np

PROCESSED = "/content/drive/MyDrive/waymo/processed"
OUT = "/content/drive/MyDrive/waymo/lidar_min_dist"
os.makedirs(OUT, exist_ok=True)

all_min_d = []

for segment in sorted(os.listdir(PROCESSED)):
    seg_path = f"{PROCESSED}/{segment}"
    if not os.path.isdir(seg_path):
        continue

    print(f"Processing {segment}...")

    for frame in sorted(os.listdir(seg_path)):
        frame_path = f"{seg_path}/{frame}"
        lidar_path = f"{frame_path}/lidar.npy"

        if not os.path.exists(lidar_path):
            continue

        raw = np.load(lidar_path, allow_pickle=True)
        points = np.array(raw, dtype=np.float32) 
        d = np.linalg.norm(points, axis=1)
        min_d = float(np.min(d))

        all_min_d.append(min_d)

np.save(f"{OUT}/lidar_min_dist.npy", np.array(all_min_d))
print("Saved:", f"{OUT}/lidar_min_dist.npy")
