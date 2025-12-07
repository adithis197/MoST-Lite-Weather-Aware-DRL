# train_td3.py
import numpy as np
import torch
from td3_agent import TD3, device
import matplotlib.pyplot as plt
import os

DATA = "/content/drive/MyDrive/waymo/td3_buffer_final"

# ---------------------- Load ----------------------
states      = np.load(f"{DATA}/states.npy")
actions     = np.load(f"{DATA}/actions.npy")
rewards     = np.load(f"{DATA}/rewards.npy")
next_states = np.load(f"{DATA}/next_states.npy")
dones       = np.load(f"{DATA}/dones.npy")

print("Loaded TD3 buffer:", states.shape, actions.shape, rewards.shape)

state_dim = states.shape[1]
action_dim = actions.shape[1]
max_action = 1.0

agent = TD3(state_dim, action_dim, max_action)

epochs = 200
batch_size = 64
updates_per_epoch = 100   

critic_losses = []
actor_losses = []
reward_curve = []

print("\n🚀 Starting TD3 Training...\n")

# ---------------------- TRAINING LOOP ----------------------
for epoch in range(epochs):

    for _ in range(updates_per_epoch):
        idx = np.random.randint(0, len(states), batch_size)

        s  = states[idx]
        a  = actions[idx]
        r  = rewards[idx]
        s2 = next_states[idx]
        d  = dones[idx]

        critic_loss, actor_loss = agent.train_step(s, a, r, s2, d)

    critic_losses.append(critic_loss)
    actor_losses.append(actor_loss)
    reward_curve.append(r.mean())

    if epoch % 10 == 0:
        print(f"Epoch {epoch}/{epochs} | Critic Loss: {critic_loss:.4f} | Actor Loss: {actor_loss:.4f}")

# ---------------------- SAVE POLICY ----------------------
model_path = f"{DATA}/td3_policy.pth"
torch.save(agent.actor.state_dict(), model_path)
print(f"\n🎉 Saved TD3 policy → {model_path}")

# ---------------------- SAVE LOSS CURVES ----------------------
np.save(f"{DATA}/critic_loss.npy", critic_losses)
np.save(f"{DATA}/actor_loss.npy", actor_losses)
np.save(f"{DATA}/reward_history.npy", reward_curve)

# ---------------------- PLOT DASHBOARD ----------------------
plt.figure(figsize=(14,7))

plt.subplot(3,1,1)
plt.plot(critic_losses, label="Critic Loss")
plt.legend(); plt.grid()

plt.subplot(3,1,2)
plt.plot(actor_losses, label="Actor Loss")
plt.legend(); plt.grid()

plt.subplot(3,1,3)
plt.plot(reward_curve, label="Average Reward")
plt.legend(); plt.grid()

plt.tight_layout()

plt.savefig(f"{DATA}/training_dashboard.png")
plt.savefig(f"{DATA}/training_dashboard.pdf")
print("\nSaved training plots (PNG & PDF).")
