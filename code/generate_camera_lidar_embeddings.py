#!/usr/bin/env python3
import os
import re
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import clip


# ============================================================
# CONFIG
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE = "/content/drive/MyDrive/waymo/processed"
OUT  = "/content/drive/MyDrive/waymo/embeddings"
os.makedirs(OUT, exist_ok=True)

print(f"Using device: {DEVICE}")

clip_model, preprocess = clip.load("ViT-B/32", device=DEVICE)

# Regex to enforce frame_000 format ONLY
FRAME_REGEX = re.compile(r"^frame_\d{3}$")


# ============================================================
# LIDAR ENCODER
# ============================================================
class LiDARMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
        )

    def forward(self, x):
        return self.net(x)


lidar_encoder = LiDARMLP().to(DEVICE)
lidar_encoder.eval()


# ============================================================
# HELPERS
# ============================================================
def encode_image(img_path):
    img = Image.open(img_path).convert("RGB")
    t = preprocess(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        f = clip_model.encode_image(t)
    return f.squeeze().cpu().numpy()


def clean_lidar(x):
    """Ensure Nx3 float32 LiDAR array from object/ragged arrays."""
    if isinstance(x, np.ndarray) and x.dtype != object and x.ndim == 2 and x.shape == (x.shape[0], 3):
        return x.astype(np.float32)

    # Object array
    if isinstance(x, np.ndarray) and x.dtype == object:
        pts = []
        for item in x:
            arr = np.array(item)
            if arr.ndim == 2 and arr.shape[1] == 3:
                pts.append(arr)
            elif arr.ndim == 1 and arr.size >= 3:
                pts.append(arr.reshape(1, 3))
        if len(pts) == 0:
            raise ValueError("Object array contained no valid LiDAR points")
        return np.concatenate(pts, axis=0).astype(np.float32)

    # Ragged fallback
    if x.ndim == 1:
        pts = [np.array(item).reshape(-1, 3) for item in x]
        return np.concatenate(pts, axis=0).astype(np.float32)

    raise ValueError(f"Unsupported LiDAR shape {x.shape}, dtype {x.dtype}")


def encode_lidar(xyz):
    if xyz.shape[0] < 5:
        xyz = np.zeros((10, 3), dtype=np.float32)

    mean = xyz.mean(axis=0)
    std = xyz.std(axis=0)

    v = np.concatenate([mean, std]).astype(np.float32)
    v = torch.tensor(v).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        f = lidar_encoder(v)

    return f.squeeze().cpu().numpy()


# ============================================================
# MAIN LOOP
# ============================================================
all_embeddings = []

print("\n==================== BEGIN EMBEDDING EXTRACTION ====================\n")

for segment in sorted(os.listdir(BASE)):
    seg_path = f"{BASE}/{segment}"
    if not os.path.isdir(seg_path):
        continue

    print(f"\nProcessing {segment} ...")

    # Only process directories like frame_000, frame_001
    frames = sorted([
        f for f in os.listdir(seg_path)
        if FRAME_REGEX.match(f) and os.path.isdir(os.path.join(seg_path, f))
    ])

    for frame in frames:
        frame_path = f"{seg_path}/{frame}"

        img_path = f"{frame_path}/front.png"
        lidar_path = f"{frame_path}/lidar.npy"
        out_path = f"{frame_path}/embedding.npy"

        if not os.path.exists(img_path) or not os.path.exists(lidar_path):
            continue

        if os.path.exists(out_path):
            emb = np.load(out_path)
            all_embeddings.append(emb)
            print(f"Skipped {frame}, already encoded.")
            continue

        # Encode image
        f_img = encode_image(img_path)

        # Load + clean lidar
        raw = np.load(lidar_path, allow_pickle=True)
        lidar_xyz = clean_lidar(raw)

        # Encode LiDAR
        f_lidar = encode_lidar(lidar_xyz)

        # Fuse -> 768D
        z = np.concatenate([f_img, f_lidar])

        np.save(out_path, z)
        all_embeddings.append(z)

        print(f"✔ {frame}: Embedded → {z.shape}")


# ============================================================
# SAVE FINAL DATASET
# ============================================================
all_embeddings = np.array(all_embeddings)
np.save(f"{OUT}/scene_embeddings.npy", all_embeddings)

print(f"Saved → {OUT}/scene_embeddings.npy")
print(f"Total frames encoded: {len(all_embeddings)}")
print("Final shape:", all_embeddings.shape)
