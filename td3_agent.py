import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

# ----- ACTOR -----
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.l2 = nn.Linear(512, 384)
        self.bn2 = nn.BatchNorm1d(384)
        self.l3 = nn.Linear(384, 256)
        self.l4 = nn.Linear(256, action_dim)
        self.dropout = nn.Dropout(0.1)
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.bn1(self.l1(state)))
        a = self.dropout(a)
        a = F.relu(self.bn2(self.l2(a)))
        a = self.dropout(a)
        a = F.relu(self.l3(a))
        a = self.l4(a)
        return self.max_action * torch.tanh(a)


# ----- CRITIC -----
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 512)
        self.l2 = nn.Linear(512, 384)
        self.l3 = nn.Linear(384, 256)
        self.l4 = nn.Linear(256, 1)
        # Q2 architecture
        self.l5 = nn.Linear(state_dim + action_dim, 512)
        self.l6 = nn.Linear(512, 384)
        self.l7 = nn.Linear(384, 256)
        self.l8 = nn.Linear(256, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        # Q1
        q1 = F.relu(self.l1(sa))
        q1 = self.dropout(q1)
        q1 = F.relu(self.l2(q1))
        q1 = self.dropout(q1)
        q1 = F.relu(self.l3(q1))
        q1 = self.l4(q1)
        # Q2
        q2 = F.relu(self.l5(sa))
        q2 = self.dropout(q2)
        q2 = F.relu(self.l6(q2))
        q2 = self.dropout(q2)
        q2 = F.relu(self.l7(q2))
        q2 = self.l8(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = F.relu(self.l3(q1))
        return self.l4(q1)


# ----- REPLAY BUFFER -----
class ReplayBuffer:
    def __init__(self, max_size=int(1e6)):
        self.ptr = 0
        self.size = 0
        self.max_size = max_size
        self.state = []
        self.action = []
        self.next_state = []
        self.reward = []
        self.done = []

    def add(self, state, action, reward, next_state, done):
        if self.size < self.max_size:
            self.state.append(state)
            self.action.append(action)
            self.reward.append(reward)
            self.next_state.append(next_state)
            self.done.append(done)
        else:
            self.state[self.ptr] = state
            self.action[self.ptr] = action
            self.reward[self.ptr] = reward
            self.next_state[self.ptr] = next_state
            self.done[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(np.array([self.state[i] for i in ind])),
            torch.FloatTensor(np.array([self.action[i] for i in ind])),
            torch.FloatTensor(np.array([self.reward[i] for i in ind])).unsqueeze(1),
            torch.FloatTensor(np.array([self.next_state[i] for i in ind])),
            torch.FloatTensor(np.array([self.done[i] for i in ind])).unsqueeze(1)
        )


# ----- TD3 AGENT -----
class TD3Agent:
    def __init__(self, state_dim, action_dim, max_action,
                 discount=0.99, tau=0.005,
                 policy_noise=0.2, noise_clip=0.5, policy_freq=2):

        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau

        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0
        self.replay_buffer = ReplayBuffer()

    def select_action(self, state, noise=0.1):
        state = torch.FloatTensor(state.reshape(1, -1))
        self.actor.eval()  # Set to eval mode for BatchNorm
        with torch.no_grad():
            action = self.actor(state).cpu().numpy()[0]
        self.actor.train()  # Set back to train mode
        if noise != 0:
            action = action + np.random.normal(0, noise, size=action.shape)
        return np.clip(action, -self.max_action, self.max_action)

    def train(self, batch_size=64):
        if self.replay_buffer.size < batch_size:
            return

        self.total_it += 1

        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        with torch.no_grad():
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * self.discount * target_Q

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed actor updates
        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Target network update
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename)
        print(f"💾 Model saved to {filename}")

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename))
        print(f"✅ Model loaded from {filename}")
