from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import random
import argparse
from evaluate import evaluate_HIV, evaluate_HIV_population
import itertools
from copy import deepcopy
import os

env = TimeLimit(
    env=HIVPatient(domain_randomization=True), max_episode_steps=200
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
    
class DQN(nn.Module):
    def __init__(self, env, hidden_size, depth):
        super(DQN, self).__init__()
        self.in_layer = nn.Linear(env.observation_space.shape[0], hidden_size)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(depth - 1)])
        self.out_layer = nn.Linear(hidden_size, env.action_space.n)

    def forward(self, x):
        x = F.relu(self.in_layer(x))
        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))
        return self.out_layer(x)

parser = argparse.ArgumentParser()
parser.add_argument('--max_episode', type=int, default=4000)
parser.add_argument('--gradient_step', type=int, default=3)
parser.add_argument('--monitoring_freq', type=int, default=50)
parser.add_argument('--monitoring_nb_trials', type=int, default=0)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--hidden_size', type=int, default=512)
parser.add_argument('--depth', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--update_target_freq', type=int, default=400)
parser.add_argument('--epsilon_decay_period', type=int, default=10000)
parser.add_argument('-epsilon_delay_decay', type=int, default=600)
parser.add_argument('--model_name', type=str, default='best_agent_12')
parser.add_argument('--delay_save', type=int, default=100)

args = parser.parse_args()
config = vars(args)

class ProjectAgent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = config['batch_size'] if 'batch_size' in config else 256
        self.gamma = config['gamma'] if 'gamma' in config else 0.98
        self.learning_rate = config['learning_rate'] if 'learning_rate' in config else 0.001
        self.nb_gradient_steps = config['nb_gradient_steps'] if 'nb_gradient_steps' in config else 2
        self.min_exploration_rate = config['min_exploration_rate'] if 'min_exploration_rate' in config else 0.01
        self.memory_capacity = config['memory_capacity'] if 'memory_capacity' in config else 100000
        self.max_episode = config['max_episode'] if 'max_episode' in config else 100
        self.memory = ReplayBuffer(self.memory_capacity, self.device)
        self.model_hidden_size = config['hidden_size'] if 'hidden_size' in config else 256
        self.model_depth = config['depth'] if 'depth' in config else 5
        self.model = self.build_model(env, hidden_size=self.model_hidden_size, depth=self.model_depth).to(self.device)
        self.target_model = deepcopy(self.model).to(self.device)
        self.update_target_strategy = config['update_target_strategy'] if 'update_target_strategy' in config else 'replace'
        self.update_target_freq = config['update_target_freq'] if 'update_target_freq' in config else 20
        self.update_target_tau = config['update_target_tau'] if 'update_target_tau' in config else 0.005
        self.criterion = config['criterion'] if 'criterion' in config else nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.epsilon_max = config['epsilon_max'] if 'epsilon_max' in config else 1.
        self.epsilon_min = config['epsilon_min'] if 'epsilon_min' in config else 0.01
        self.epsilon_stop = config['epsilon_stop'] if 'epsilon_stop' in config else 10000
        self.epsilon_delay = config['epsilon_delay_decay'] if 'epsilon_delay_decay' in config else 600
        self.epsilon_step = (self.epsilon_max - self.epsilon_min) / self.epsilon_stop
        self.monitoring_nb_trials = config['monitoring_nb_trials'] if 'monitoring_nb_trials' in config else 0
        self.monitoring_freq = config['monitoring_freq'] if 'monitoring_freq' in config else 50
        self.model_name = config['model_name'] if 'model_name' in config else 'best_agent'
        self.delay_save = config['delay_save'] if 'delay_save' in config else 100
    
    def build_model(self, env, hidden_size, depth):
        return DQN(env, hidden_size, depth)
    
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
            return env.action_space.sample()
        else:
            return self.greedy_action(self.model, observation)
    
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
        best_score = 0
        while episode < max_episode:
            # update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)
            # select epsilon-greedy action
            if np.random.rand() < epsilon:
                # action = env.action_space.sample()
                action = self.act(state, use_random=True)
            else:
                # action = self.greedy_action(self.model, state)
                action = self.act(state, use_random=False)
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
                if self.monitoring_nb_trials>0 and episode % self.monitoring_freq == 0:
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
                        
                    # if MC_tr > best_score:
                    #     best_score = MC_tr
                    #     self.save("best_agent.pth")
                    
                else:
                    episode_return.append(episode_cum_reward)
                    print("Episode ", '{:2d}'.format(episode), 
                          ", epsilon ", '{:6.2f}'.format(epsilon), 
                          ", batch size ", '{:4d}'.format(len(self.memory)), 
                          ", ep return ", '{:4.1f}'.format(episode_cum_reward), 
                          sep='')
                    
                # Evaluation
                score_agent = evaluate_HIV(agent=self, nb_episode=1)
                if (episode > self.delay_save) and (score_agent > best_score):
                    best_score = score_agent
                    self.save(f"{os.getcwd()}/" + self.model_name + '.pth')
                    print("Best score updated: ", "{:e}".format(best_score))
                
                state, _ = env.reset()
                episode_cum_reward = 0
            else:
                state = next_state
        
        return episode_return, MC_avg_discounted_reward, MC_avg_total_reward, V_init_state
    
    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self):
        self.model.load_state_dict(torch.load(f"{os.getcwd()}/src/" + self.model_name + '.pth', map_location='cpu'))
        self.model.eval()

if __name__ == "__main__":

    agent = ProjectAgent()
    episode_return, MC_avg_discounted_reward, MC_avg_total_reward, V_init_state = agent.train(env, config['max_episode'])
    
    # score_agent = evaluate_HIV(agent=agent, nb_episode=1)
    # score_agent_dr = evaluate_HIV_population(agent=agent, nb_episode=15)
    
    print("agent name:" + config['model_name'] + ".pth")
    print('config:', config)
    
    # print("score_agent:" + "{:e}".format(score_agent))
    # print("score_agent_dr:" + "{:e}".format(score_agent_dr))
    
    