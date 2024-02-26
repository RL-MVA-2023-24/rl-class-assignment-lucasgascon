from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import random
from evaluate import evaluate_HIV, evaluate_HIV_population
import itertools
from sklearn.ensemble import RandomForestRegressor
from copy import deepcopy

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
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    def __len__(self):
        return len(self.data)

class ProjectAgent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 256
        self.gamma = 0.98 
        self.length_episode = 200
        self.learning_rate = 0.001
        self.nb_gradient_steps = 3
        self.min_exploration_rate = 0.01
        self.memory_capacity = 100000
        self.max_episode = 100
        self.memory = ReplayBuffer(self.memory_capacity, self.device)
        self.model_hidden_size = 256
        self.model_depth = 10
        self.model = self.build_model(env, hidden_size=self.model_hidden_size, depth = self.model_depth)
        self.target_model = deepcopy(self.model).to(self.device)
        self.update_target_strategy = 'replace'
        self.update_target_freq = 20
        self.update_target_tau = 0.005
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.epsilon_max = 1.
        self.epsilon_min = 0.01
        self.epsilon_stop = 10000
        self.epsilon_delay = 1000
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
        self.monitoring_nb_trials = 0
        
    # def build_model(self, env, hidden_size=32):
    #     DQN = nn.Sequential(
    #         nn.Linear(env.observation_space.shape[0], hidden_size),
    #         nn.ReLU(),
    #         nn.Linear(hidden_size, hidden_size),
    #         nn.ReLU(),
    #         nn.Linear(hidden_size, env.action_space.n)
    #     )
    #     return DQN.to(self.device)
    
    def build_model(env, hidden_size, depth):
        in_layer = nn.Linear(env.observation_space.shape[0], hidden_size)
        hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(depth - 1)])
        out_layer = nn.Linear(hidden_size, env.action_space.n)

        def forward(x):
            x = nn.ReLU(in_layer(x))
            for hidden_layer in hidden_layers:
                x = nn.ReLU(hidden_layer(x))
            return out_layer(x)

        return forward
    
    def greedy_action(self, network, state):
        with torch.no_grad():
            Q = network(torch.Tensor(state).unsqueeze(0).to(self.device))
            return torch.argmax(Q).item()
    
    def MC_eval(self, env, nb_trials):   # NEW NEW NEW
        MC_total_reward = []
        MC_discounted_reward = []
        for _ in range(nb_trials):
            x,_ = env.reset()
            done = False
            trunc = False
            total_reward = 0
            discounted_reward = 0
            step = 0
            while not (done or trunc):
                a = self.greedy_action(self.model, x)
                y,r,done,trunc,_ = env.step(a)
                x = y
                total_reward += r
                discounted_reward += self.gamma**step * r
                step += 1
            MC_total_reward.append(total_reward)
            MC_discounted_reward.append(discounted_reward)
        return np.mean(MC_discounted_reward), np.mean(MC_total_reward)
    
    def V_initial_state(self, env, nb_trials):   # NEW NEW NEW
        with torch.no_grad():
            for _ in range(nb_trials):
                val = []
                x,_ = env.reset()
                val.append(self.model(torch.Tensor(x).unsqueeze(0).to(self.device)).max().item())
        return np.mean(val)
    
    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.model(Y).max(1)[0].detach()
            #update = torch.addcmul(R, self.gamma, 1-D, QYmax)
            update = torch.addcmul(R, 1 - D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
    def act(self, observation, use_random=False):
        if use_random:
            return np.random.choice(4)
        Qsa = []
        for a in range(env.action_space.n):
            sa = np.append(observation,a).reshape(1, -1)
            Qsa.append(self.Q.predict(sa))
        action = np.argmax(Qsa)
        return action
    
    def train(self, env, max_episode):
        episode_return = []
        MC_avg_total_reward = []   # NEW NEW NEW
        MC_avg_discounted_reward = []   # NEW NEW NEW
        V_init_state = []   # NEW NEW NEW
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0
        while episode < max_episode:
            # update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)
            # select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = self.greedy_action(self.model, state)
            # step
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward
            # train
            for _ in range(self.nb_gradient_steps): 
                self.gradient_step()
            # update target network if needed
            if self.update_target_strategy == 'replace':
                if step % self.update_target_freq == 0: 
                    self.target_model.load_state_dict(self.model.state_dict())
            if self.update_target_strategy == 'ema':
                target_state_dict = self.target_model.state_dict()
                model_state_dict = self.model.state_dict()
                tau = self.update_target_tau
                for key in model_state_dict:
                    target_state_dict[key] = tau*model_state_dict + (1-tau)*target_state_dict
                self.target_model.load_state_dict(target_state_dict)
            # next transition
            step += 1
            if done or trunc:
                episode += 1
                # Monitoring
                if self.monitoring_nb_trials>0:
                    MC_dr, MC_tr = self.MC_eval(env, self.monitoring_nb_trials)    # NEW NEW NEW
                    V0 = self.V_initial_state(env, self.monitoring_nb_trials)   # NEW NEW NEW
                    MC_avg_total_reward.append(MC_tr)   # NEW NEW NEW
                    MC_avg_discounted_reward.append(MC_dr)   # NEW NEW NEW
                    V_init_state.append(V0)   # NEW NEW NEW
                    episode_return.append(episode_cum_reward)   # NEW NEW NEW
                    print("Episode ", '{:2d}'.format(episode), 
                          ", epsilon ", '{:6.2f}'.format(epsilon), 
                          ", batch size ", '{:4d}'.format(len(self.memory)), 
                          ", ep return ", '{:4.1f}'.format(episode_cum_reward), 
                          ", MC tot ", '{:6.2f}'.format(MC_tr),
                          ", MC disc ", '{:6.2f}'.format(MC_dr),
                          ", V0 ", '{:6.2f}'.format(V0),
                          sep='')
                else:
                    episode_return.append(episode_cum_reward)
                    print("Episode ", '{:2d}'.format(episode), 
                          ", epsilon ", '{:6.2f}'.format(epsilon), 
                          ", batch size ", '{:4d}'.format(len(self.memory)), 
                          ", ep return ", '{:4.1f}'.format(episode_cum_reward), 
                          sep='')

                
                state, _ = env.reset()
                episode_cum_reward = 0
            else:
                state = next_state
        return episode_return, MC_avg_discounted_reward, MC_avg_total_reward, V_init_state
    
    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self):
        self.model.load_state_dict(torch.load("agent.pth", map_location='cpu'))
        self.model.eval()

if __name__ == "__main__":
    agent = ProjectAgent()
    episode_return, MC_avg_discounted_reward, MC_avg_total_reward, V_init_state = agent.train(env, 100)
    agent.save("agent.pth")