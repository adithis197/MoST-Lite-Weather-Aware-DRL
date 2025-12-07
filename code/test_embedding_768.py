import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import clip

# ============================================
# CONFIG
# ============================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_PATH = "/content/drive/MyDrive/waymo/processed/segment_0/frame_000/front.png"
LIDAR_PATH = "/content/drive/MyDrive/waymo/processed/segment_0/frame_000/lidar.npy"

print(f"Using device: {DEVICE}")

# Load CLIP
clip_model, preprocess = clip.load("ViT-B/32", device=DEVICE)

# ============================================
# LiDAR Encoder (Simple MLP)
# ============================================
class LiDARMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256)
        )

    def forward(self, x):
        return self.net(x)

lidar_encoder = LiDARMLP().to(DEVICE)
lidar_encoder.eval()

# ============================================
# FUNCTIONS
# ============================================
def encode_image(path):
    img = Image.open(path).convert("RGB")
    t = preprocess(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        f = clip_model.encode_image(t)
    return f.squeeze().cpu().numpy()  # (512,)


def encode_lidar(xyz):
    """xyz must be Nx3 float array"""
    if xyz.shape[0] < 10:
        raise ValueError("LiDAR has too few points.")

    mean = xyz.mean(axis=0)
    std  = xyz.std(axis=0)

    v = np.concatenate([mean, std]).astype(np.float32)   # (6,)
    v = torch.tensor(v).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        f = lidar_encoder(v)

    return f.squeeze().cpu().numpy()  # (256,)


# ============================================
# MAIN
# ============================================
print("\nLoading frame...")

# Load image → embedding
img_feat = encode_image(IMG_PATH)
print("Image embedding:", img_feat.shape)

# Load LiDAR → embedding
lidar = np.load(LIDAR_PATH)
if lidar.ndim != 2 or lidar.shape[1] != 3:
    raise ValueError(f"Invalid LiDAR shape: {lidar.shape}")

lidar_feat = encode_lidar(lidar)
print("LiDAR embedding:", lidar_feat.shape)

# Fuse → 768-D
z = np.concatenate([img_feat, lidar_feat])
print("\nFinal fused embedding shape:", z.shape)

# Save
np.save("single_frame_embedding.npy", z)
print("Saved → single_frame_embedding.npy")
