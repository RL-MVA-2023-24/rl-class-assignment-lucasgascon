from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import random

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!


class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled
        self.device = device
    def append(self, s, a, r, s_):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_)
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    def __len__(self):
        return len(self.data)

class DQN(nn.Module):
    def __init__(self, state_dim, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class DQN2(nn.Module):
    def __init__(self, state_dim, action_size):
        super(DQN2, self).__init__()
        # Increase the depth and width of the network
        self.fc1 = nn.Linear(state_dim, 64)  # Increased width
        self.fc2 = nn.Linear(64, 128)  # Increased width and added a layer
        self.fc3 = nn.Linear(128, 64)  # Added another layer
        self.fc4 = nn.Linear(64, action_size)  # Final layer for action sizes
        
        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))  # Using LeakyReLU for activation
        x = self.dropout(x)  # Apply dropout
        x = F.leaky_relu(self.fc2(x))  # Using LeakyReLU for activation
        x = self.dropout(x)  # Apply dropout
        x = F.leaky_relu(self.fc3(x))  # Using LeakyReLU for activation
        x = self.fc4(x)  # No activation here to allow for raw output for Q-values
        return x


class ProjectAgent:
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = 6
        self.action_size = 4
        self.batch_size = 200
        self.gamma = 0.99
        self.length_episode = 100  # The time wrapper limits the number of steps in an episode at 200.
        self.learning_rate = 0.005
        self.exploration_rate = 1.0
        self.exploration_decay = 0.995
        self.min_exploration_rate = 0.01
        self.memory = ReplayBuffer(10000, device)
        self.max_episode = 100
        self.model = DQN2(self.state_dim, self.action_size).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
    
    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y = self.memory.sample(self.batch_size)
            QYmax = self.model(Y).max(1)[0].detach()
            #update = torch.addcmul(R, self.gamma, 1-D, QYmax)
            update = torch.addcmul(R, torch.ones(1), QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 
    
    def act(self, observation, use_random=False):
        if np.random.rand() < self.exploration_rate:
            return np.random.choice(self.action_size)
        observation = torch.tensor(observation, dtype=torch.float32)
        q_values = self.model(observation)
        return torch.argmax(q_values).item()
    
    def train(self, env, max_episode):
        episode_return = []
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        step = 0

        for episode in tqdm(range(max_episode)):
            for step in range(self.length_episode):
                # Update exploration rate
                self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)
                
                # select epsilon-greedy action
                action = self.act(state)

                # step
                next_state, reward, _, _, _ = env.step(action)
                self.memory.append(state, action, reward, next_state)
                episode_cum_reward += reward

                # train
                self.gradient_step()
                
                # update state
                state = next_state
                    
            print("Episode ", '{:3d}'.format(episode), 
                ", exploration rate ", '{:6.2f}'.format(self.exploration_rate), 
                ", batch size ", '{:5d}'.format(len(self.memory)), 
                ", episode return ", '{:4.1f}'.format(episode_cum_reward),
                sep='')
            state, _ = env.reset()
            episode_return.append(episode_cum_reward)
            episode_cum_reward = 0

        return episode_return
        
    def save(self, path):
        torch.save(self.model, path)

    def load(self):
        self.train(env, self.max_episode)
        return torch.load("model.pth")