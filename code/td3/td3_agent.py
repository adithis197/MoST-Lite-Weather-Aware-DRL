# td3_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------- Actor ----------------------
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return self.max_action * torch.tanh(self.l3(x))

# ---------------------- Critic (Q1 & Q2) ----------------------
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        # Q1
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        # Q2
        self.l4 = nn.Linear(state_dim + action_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)

    def forward(self, x, u):
        xu = torch.cat([x, u], dim=1)

        q1 = torch.relu(self.l1(xu))
        q1 = torch.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = torch.relu(self.l4(xu))
        q2 = torch.relu(self.l5(q2))
        q2 = self.l6(q2)

        return q1, q2

    def q1(self, x, u):
        xu = torch.cat([x, u], dim=1)
        q1 = torch.relu(self.l1(xu))
        q1 = torch.relu(self.l2(q1))
        return self.l3(q1)


# ---------------------- TD3 AGENT ----------------------
class TD3:
    def __init__(self, state_dim, action_dim, max_action):

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.opt_actor = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.discount = 0.99
        self.tau = 0.005
        self.policy_freq = 2
        self.total_it = 0

    # ---------------------- TRAIN STEP (FIXED) ----------------------
    def train_step(self, state, action, reward, next_state, done):
        self.total_it += 1

        state      = torch.FloatTensor(state).to(device)
        action     = torch.FloatTensor(action).to(device)
        reward     = torch.FloatTensor(reward).unsqueeze(1).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        done       = torch.FloatTensor(done).unsqueeze(1).to(device)

        # -------- Target Policy Smoothing --------
        noise = (torch.randn_like(action) * 0.2).clamp(-0.5, 0.5)
        next_action = (self.actor_target(next_state) + noise).clamp(-1, 1)

        # -------- Target Q-Value --------
        target_q1, target_q2 = self.critic_target(next_state, next_action)
        target_q = reward + (1 - done) * self.discount * torch.min(target_q1, target_q2)

        # -------- Critic Update --------
        current_q1, current_q2 = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_q1, target_q.detach()) + nn.MSELoss()(current_q2, target_q.detach())

        self.opt_critic.zero_grad()
        critic_loss.backward()
        self.opt_critic.step()

        # -------- Actor Update (EVERY 2 STEPS) --------
        actor_loss = None
        if self.total_it % self.policy_freq == 0:
            pi = self.actor(state)
            actor_loss = -self.critic.q1(state, pi).mean()

            self.opt_actor.zero_grad()
            actor_loss.backward()
            self.opt_actor.step()

            # Soft update
            for p, p_targ in zip(self.actor.parameters(), self.actor_target.parameters()):
                p_targ.data.copy_(self.tau * p.data + (1 - self.tau) * p_targ.data)

            for p, p_targ in zip(self.critic.parameters(), self.critic_target.parameters()):
                p_targ.data.copy_(self.tau * p.data + (1 - self.tau) * p_targ.data)

        return critic_loss.item(), (actor_loss.item() if actor_loss is not None else 0.0)
