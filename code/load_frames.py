#!/usr/bin/env python3
import os
import tensorflow as tf
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import frame_utils
import numpy as np
from PIL import Image

# --------------------------------------------------
# PATHS
# --------------------------------------------------
BASE_PATH = "/content/drive/MyDrive/waymo/train_segments"
OUT_PATH = "/content/drive/MyDrive/waymo/processed"
os.makedirs(OUT_PATH, exist_ok=True)

files = sorted([f for f in os.listdir(BASE_PATH) if f.endswith(".tfrecord")])
assert len(files) > 0, "No TFRecords found!"

print(f"Found {len(files)} TFRecord segments")
print("-" * 60)

# --------------------------------------------------
# MAIN LOOP
# --------------------------------------------------
for file_i, fname in enumerate(files):
    seg_path = f"{BASE_PATH}/{fname}"
    print(f"\nLoading segment {file_i+1}/{len(files)} → {fname}")

    segment_out = f"{OUT_PATH}/segment_{file_i}"
    os.makedirs(segment_out, exist_ok=True)

    dataset = tf.data.TFRecordDataset(seg_path, compression_type='')

    for frame_i, data in enumerate(dataset):
        frame = open_dataset.Frame()
        frame.ParseFromString(data.numpy())

        frame_dir = f"{segment_out}/frame_{frame_i}"
        os.makedirs(frame_dir, exist_ok=True)

        # --------------------------------------------------
        # CAMERA: extract FRONT image
        # --------------------------------------------------
        front_img = None
        for img in frame.images:
            if img.name == open_dataset.CameraName.FRONT:
                front_img = tf.io.decode_jpeg(img.image).numpy()
                break

        if front_img is not None:
            Image.fromarray(front_img).save(f"{frame_dir}/front.png")
        else:
            print(f"⚠️ Frame {frame_i}: FRONT camera missing")
            continue

        # --------------------------------------------------
        # LIDAR
        # --------------------------------------------------
        try:
            parsed = frame_utils.parse_range_image_and_camera_projection(frame)
            if len(parsed) == 5:
                points, cp, ri, cp_points, seg_labels = parsed
            else:
                points, cp, ri, cp_points = parsed
        except Exception as e:
            print(f"Error parsing LiDAR for frame {frame_i}: {e}")
            continue

        pc_list = []
        for laser_id in points:
            pc_list.append(points[laser_id])

        if len(pc_list) > 0:
            xyz = np.concatenate(pc_list, axis=0)
            np.save(f"{frame_dir}/lidar.npy", xyz)
        else:
            print(f"Frame {frame_i}: No LiDAR points")

        print(f"Frame {frame_i} saved → {frame_dir}")

print("\n DONE! All segments processed successfully.")
