#!/usr/bin/env python3
import os
import pickle
import numpy as np
import tensorflow as tf
from waymo_open_dataset import dataset_pb2 as open_dataset

RAW = "/content/drive/MyDrive/waymo/train_segments"
OUT = "/content/drive/MyDrive/waymo/metadata"
os.makedirs(OUT, exist_ok=True)

def extract_pose(frame):
    """Extract ego motion metadata safely."""
    p = frame.pose
    
    # Translation (position)
    x = p.transform[3]
    y = p.transform[7]
    z = p.transform[11]

    # Yaw angle
    yaw = float(np.arctan2(p.transform[4], p.transform[0]))

    # Safe attributes
    speed = getattr(p, "speed", 0.0)
    yaw_rate = getattr(p, "angular_velocity", 0.0)

    return x, y, z, yaw, speed, yaw_rate


print("Extracting metadata from TFRecords...\n")

all_segments = []

segments = sorted([f for f in os.listdir(RAW) if f.endswith(".tfrecord")])

for sid, fname in enumerate(segments):

    tf_path = os.path.join(RAW, fname)
    print(f"\n📂 Segment {sid}: {fname}")

    dataset = tf.data.TFRecordDataset(tf_path)

    timestamps = []
    positions = []
    yaws = []
    speeds = []
    yaw_rates = []

    for record in dataset:
        frame = open_dataset.Frame()
        frame.ParseFromString(record.numpy())

        timestamps.append(frame.timestamp_micros / 1e6)

        x, y, z, yaw, speed, yaw_rate = extract_pose(frame)

        positions.append((x, y, z))
        yaws.append(yaw)
        speeds.append(speed)
        yaw_rates.append(yaw_rate)

    # Convert to numpy
    seg_meta = {
        "timestamps": np.array(timestamps),
        "positions": np.array(positions),
        "yaw": np.array(yaws),
        "speed": np.array(speeds),
        "yaw_rate": np.array(yaw_rates),
    }

    print(f"Collected {len(timestamps)} frames.")

    all_segments.append(seg_meta)

# SAVE OUTPUT
out_path = f"{OUT}/metadata_correct.pkl"
with open(out_path, "wb") as f:
    pickle.dump(all_segments, f)

print("\nDONE — Saved metadata at:", out_path)
print("Total segments saved:", len(all_segments))
