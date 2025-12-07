# Weather-Aware Autonomous Driving with TD3

Reinforcement Learning over Multi-Modal Waymo Scene Embeddings

This repository implements a **weather-aware autonomous driving controller** using **Twin Delayed Deep Deterministic Policy Gradients (TD3)** trained on **multi-modal scene representations** extracted from the **Waymo Open Dataset**.

The project is inspired by ideas from the **MOST: Multi-Modal Scene Tokenization** paper from Waymo, showing that combining vision, LiDAR, and environment context improves downstream planning and control.

Paper link:
[https://arxiv.org/abs/2404.19531](https://arxiv.org/abs/2404.19531)

---

## 1. Waymo Dataset

Waymo Open Dataset:
[https://waymo.com/open/](https://waymo.com/open/)

Download TFRecord sequences containing:

* Multi-camera images
* LiDAR sweeps
* Calibration and pose metadata

Your structure should look like:

```
waymo/
   raw/
   frames/
   lidar/
   metadata/
   embeddings/
   td3_buffer_final/
```

---

## 2. Environment Setup

The project uses Python 3.10, TensorFlow 2.11 (Waymo-compatible), and PyTorch for TD3.

Run:

```bash
$ sudo apt-get update
$ sudo apt-get install python3.10 python3.10-venv python3.10-distutils

$ python3.10 -m venv waymo310
$ source waymo310/bin/activate

$ pip install tensorflow==2.11.0
$ pip install waymo-open-dataset-tf-2-11-0

$ pip install torch torchvision torchaudio
$ pip install matplotlib imageio opencv-python
```

---

## 3. Data Preprocessing Pipeline

All steps must be executed **in order**.
Each step corresponds to a script in the `code/` directory.

---

### 3.1 Extract Camera Frames

Script: `extract_frames.py`

```bash
$ python extract_frames.py
```

Converts TFRecord segments into RGB frames used for ViT embeddings.

---

### 3.2 Compute LiDAR Minimum Distances

Script: `extract_lidar_min_dist.py`

```bash
$ python extract_lidar_min_dist.py
```

Computes the **minimum 3D point distance** for each frame.
This single scalar is an important safety feature.

---

### 3.3 Extract Vehicle Metadata

Script: `extract_metadata.py`

```bash
$ python extract_metadata.py
```

Produces `metadata_correct.pkl` containing:

* timestamps
* (x, y, z) positions
* yaw
* speed
* yaw_rate

Used later for acceleration and steering estimation.

---

### 3.4 Generate Multi-Modal Scene Embeddings

Script: `generate_camera_lidar_embeddings.py`

```bash
$ python generate_camera_lidar_embeddings.py
```

This step performs:

1. ViT-B/16 feature extraction
2. CLIP cosine similarity for weather prompts
3. One-hot weather encoding:

   * clear
   * fog
   * rain
   * night
   * snow
4. Fusion into a final **773-dim embedding vector**
5. Saves `scene_embeddings_with_weather.npy`

---

## 4. Reinforcement Learning Pipeline (TD3)

---

### 4.1 Build the TD3 Replay Buffer

Script: `build_td3_buffer.py`

```bash
$ python build_td3_buffer.py
```

This script:

* Loads embeddings
* Loads LiDAR distances
* Loads metadata (speed, yaw rate)
* Generates **synthetic expert actions** (`build_synthetic_expert.py`)
* Normalizes/standardizes features
* Computes rewards via `reward_function.py`
* Saves replay buffer:

```
states.npy
next_states.npy
actions.npy
rewards.npy
dones.npy
```

Stored in: `td3_buffer_final/`

---

### 4.2 Train TD3

Script: `train_td3.py`

```bash
$ python train_td3.py
```

Outputs:

* `td3_policy.pth`
* `critic_loss.npy`, `actor_loss.npy`, `reward_history.npy`
* `training_dashboard.png`
* `training_dashboard.pdf`

---

### 4.3 Evaluate TD3

Script: `evaluate_td3.py`

```bash
$ python evaluate_td3.py
```

Computes:

* MSE between predicted and expert actions
* MAE
* Example predicted vs. expert actions

---

### 4.4 Visualize Policy Behavior (Toy Driving Simulator)

Script: `visualize_agent.py`

```bash
$ python visualize_agent.py
```

Creates:

```
td3_buffer_final/demo.mp4
```

This video illustrates how the learned TD3 policy responds under simplified weather and obstacle conditions.

---

## 5. State and Action Representation

### State Vector (dimension = 776)

* 768-dim ViT embedding (z-score normalized)
* 1 × LiDAR minimum distance
* 1 × ego speed
* 1 × yaw rate
* 5-dim one-hot weather vector
  (clear, fog, rain, night, snow)

### Action Vector (continuous)

* acceleration
* steering rate

---

## 6. Reward Function Summary

Encourages:

* safer driving
* slower speeds in poor visibility
* smoother motion

Penalizes:

* high speed in fog, rain, night
* steering while in bad weather
* proximity to nearby obstacles

Overall, the reward is designed to bias the agent toward conservative behavior in adverse conditions while maintaining forward motion in clear conditions.

---

## 7. Repository Structure

```
code/
   build_synthetic_expert.py
   build_td3_buffer.py
   evaluate_td3.py
   reward_function.py
   td3_agent.py
   td3_networks.py
   train_td3.py
   visualize_agent.py
   extract_frames.py
   extract_lidar_min_dist.py
   extract_metadata.py
   generate_camera_lidar_embeddings.py

embeddings/
   scene_embeddings_with_weather.npy

lidar_min_dist/
metadata/

models/
results/

td3_buffer_final/
   states.npy
   next_states.npy
   actions.npy
   rewards.npy
   dones.npy
   td3_policy.pth
   training_dashboard.png
```

---

## 8. Summary

This project demonstrates a complete RL pipeline combining:

* Image representation (ViT)
* LiDAR geometry
* Weather classification (CLIP)
* Vehicle dynamics (speed, yaw rate)
* Synthetic expert demonstrations
* TD3 continuous-control learning

The resulting agent learns **weather-aware autonomous driving behavior** on top of real-world multi-modal perception data.



