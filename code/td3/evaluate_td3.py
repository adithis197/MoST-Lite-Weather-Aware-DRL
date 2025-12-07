# evaluate_td3.py
import numpy as np
import torch
from td3_agent import Actor, device

DATA = "/content/drive/MyDrive/waymo/td3_buffer_final"
POLICY = f"{DATA}/td3_policy.pth"

states  = np.load(f"{DATA}/states.npy")
actions = np.load(f"{DATA}/actions.npy")

state_dim = states.shape[1]
action_dim = actions.shape[1]
max_action = 1.0

actor = Actor(state_dim, action_dim, max_action).to(device)
actor.load_state_dict(torch.load(POLICY, map_location=device))
actor.eval()

preds = []
for i in range(100):
    s = torch.FloatTensor(states[i]).unsqueeze(0).to(device)
    a = actor(s).cpu().detach().numpy()[0]
    preds.append(a)

mse = ((actions[:100] - preds) ** 2).mean()
mae = np.abs(actions[:100] - preds).mean()

print("✔ Loaded TD3 policy!\n")
print("TD3 Evaluation:")
print(f"  MSE: {mse:.4f}")
print(f"  MAE: {mae:.4f}\n")

for i in range(5):
    print(f"Frame {i}:")
    print("  Predicted:", preds[i])
    print("  True:     ", actions[i])
    print()
