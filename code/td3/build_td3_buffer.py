# build_td3_buffer.py
import numpy as np
import pickle
import os
from build_synthetic_expert import generate_expert_actions
from reward_function import compute_reward

EMB = "/content/drive/MyDrive/waymo/embeddings/scene_embeddings_with_weather.npy"
LIDAR = "/content/drive/MyDrive/waymo/lidar_min_dist/lidar_min_dist.npy"
META = "/content/drive/MyDrive/waymo/metadata/metadata_correct.pkl"
OUT  = "/content/drive/MyDrive/waymo/td3_buffer_final"

os.makedirs(OUT, exist_ok=True)

# ---------------------- LOAD ----------------------
emb = np.load(EMB)
lidar = np.load(LIDAR)
N = emb.shape[0]

with open(META, "rb") as f:
    meta = pickle.load(f)

seg0 = meta[0]
speed = np.array(seg0["speed"])
yaw_rate = np.array(seg0["yaw_rate"])

speed = np.pad(speed, (0, N - len(speed)), constant_values=speed[-1])
yaw_rate = np.pad(yaw_rate, (0, N - len(yaw_rate)), constant_values=0)

scene_vec = emb[:, :-5]
weather_logits = emb[:, -5:]

# ---------------------- RAW STATE ----------------------
states_raw = np.concatenate([
    scene_vec,
    lidar.reshape(-1,1),
    speed.reshape(-1,1),
    yaw_rate.reshape(-1,1),
    weather_logits
], axis=1)

# ---------------------- EXPERT ACTIONS ----------------------
actions, weather_onehot = generate_expert_actions(states_raw)

# Replace logits in state with 1-hot encoded 5D
states = np.concatenate([
    scene_vec,
    lidar.reshape(-1,1),
    speed.reshape(-1,1),
    yaw_rate.reshape(-1,1),
    weather_onehot
], axis=1)

# ---------------------- NORMALIZATION ----------------------
scene = states[:, :768]
lidar_n = states[:, 768] / 50.0
speed_n = states[:, 769] / 10.0
yaw_n   = states[:, 770]
weather = states[:, 771:]

# Normalize scene embeddings
scene_mean = scene.mean(axis=0)
scene_std = scene.std(axis=0) + 1e-6
scene_norm = (scene - scene_mean) / scene_std

states_norm = np.concatenate([
    scene_norm,
    lidar_n.reshape(-1,1),
    speed_n.reshape(-1,1),
    yaw_n.reshape(-1,1),
    weather
], axis=1)

states = states_norm

# ---------------------- TD3 TRANSITIONS ----------------------
td3_states = states[:-1]
td3_next   = states[1:]
td3_actions = actions[:-1]
td3_dones = np.zeros(len(td3_states))

# ---------------------- REWARDS ----------------------
rewards = []

for i in range(len(td3_states)):
    lidar_i  = td3_states[i][768]
    speed_i  = td3_states[i][769]
    accel_i  = td3_actions[i][0]
    steer_i  = td3_actions[i][1]
    weather_i = td3_states[i][-5:]
    r = compute_reward(lidar_i, speed_i, accel_i, steer_i, weather_i)
    rewards.append(r)

rewards = np.array(rewards)

# ---------------------- SAVE ----------------------
np.save(f"{OUT}/states.npy", td3_states)
np.save(f"{OUT}/actions.npy", td3_actions)
np.save(f"{OUT}/rewards.npy", rewards)
np.save(f"{OUT}/next_states.npy", td3_next)
np.save(f"{OUT}/dones.npy", td3_dones)

print("DONE — TD3 buffer built")
