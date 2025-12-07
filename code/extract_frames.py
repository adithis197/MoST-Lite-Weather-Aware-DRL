#!/usr/bin/env python3
import os
import tensorflow as tf
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import frame_utils
import numpy as np
from PIL import Image

RAW = "/content/drive/MyDrive/waymo/train_segments"
OUT = "/content/drive/MyDrive/waymo/processed"
os.makedirs(OUT, exist_ok=True)

def get_lidar_xyz(frame):
    range_images, camera_projections, seg_labels, top_pose = \
        frame_utils.parse_range_image_and_camera_projection(frame)

    points, cp_points = frame_utils.convert_range_image_to_point_cloud(
        frame,
        range_images,
        camera_projections,
        top_pose
    )

    pc = np.concatenate(points, axis=0).astype(np.float32)
    return pc[:, :3]

segments = sorted(os.listdir(RAW))

for seg_id, fname in list(enumerate(segments))[2:]:
    if not fname.endswith(".tfrecord"):
        continue

    tf_path = os.path.join(RAW, fname)
    seg_out = f"{OUT}/segment_{seg_id}"
    os.makedirs(seg_out, exist_ok=True)

    print(f"\n=== Processing {fname} ===")
    dataset = tf.data.TFRecordDataset(tf_path)

    for frame_id, record in enumerate(dataset):
        frame = open_dataset.Frame()
        frame.ParseFromString(record.numpy())

        frame_dir = f"{seg_out}/frame_{frame_id:03d}"
        os.makedirs(frame_dir, exist_ok=True)

        # FRONT camera
        for img in frame.images:
            if img.name == open_dataset.CameraName.FRONT:
                img_np = tf.io.decode_jpeg(img.image).numpy()
                Image.fromarray(img_np).save(f"{frame_dir}/front.png")

        # LiDAR XYZ
        pc = get_lidar_xyz(frame)
        np.save(f"{frame_dir}/lidar.npy", pc)

        print(f"Frame {frame_id}: saved image + {pc.shape[0]} lidar points")

print("\nDONE — correct LiDAR data extracted")
