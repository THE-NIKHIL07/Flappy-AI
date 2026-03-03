import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

class DuelingDQN(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(7, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU()
        )
        self.value = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.advantage = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

    def forward(self, x):
        x = self.fc(x)
        value = self.value(x)
        adv = self.advantage(x)
        return value + adv - adv.mean(dim=1, keepdim=True)

class Agent:
    def __init__(self, n_actions, lr=1e-3, gamma=0.99, buffer_size=50000, batch_size=256):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.buffer = deque(maxlen=buffer_size)

        self.net = DuelingDQN(n_actions).to(self.device)
        self.target_net = DuelingDQN(n_actions).to(self.device)
        self.target_net.load_state_dict(self.net.state_dict())
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)
        state_t = torch.tensor([state], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return self.net(state_t).argmax(1).item()

    def append(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states_t = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # Double DQN
        with torch.no_grad():
            next_actions = self.net(next_states_t).argmax(1, keepdim=True)
            next_q = self.target_net(next_states_t).gather(1, next_actions).squeeze()
            target_q = rewards_t + self.gamma * next_q * (1 - dones_t)

        current_q = self.net(states_t).gather(1, actions_t).squeeze()
        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def soft_update(self, tau=0.01):
        for target_param, param in zip(self.target_net.parameters(), self.net.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)