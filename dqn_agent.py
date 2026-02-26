import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
from dqn_network import DQNNetwork

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
  def __init__(self, capacity=10000):
    self.memory = deque(maxlen=capacity)

  def push(self, state, action, reward, next_state, done):
    self.memory.append(Transition(state, action, reward, next_state, done))

  def sample(self, batch_size):
    transitions = random.sample(self.memory, batch_size)
    batch = Transition(*zip(*transitions))
    return batch
  
  def __len__(self):
    return len(self.memory)
  

class DQNAgent:
  def __init__(self, num_actions=6, device="cpu"):
    self.num_actions = num_actions
    self.device = device
    self.gamma = 0.99
    self.epsilon = 1.0
    self.epsilon_end = 0.1
    self.epsilon_decay = 0.995

    self.network = DQNNetwork(input_shape=(4, 84, 84), num_actions=num_actions).to(device)
    self.target_network = DQNNetwork(input_shape=(4, 84, 84), num_actions=num_actions).to(device)
    self.target_network.load_state_dict(self.network.state_dict())

    self.optimizer = optim.Adam(self.network.parameters(), lr=1e-4)
    self.replay_buffer = ReplayBuffer()

  def select_action(self, state):
    if np.random.random() < self.epsilon:
      return random.randint(0, self.num_actions - 1)
    else:
      with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.network(state_tensor)
        return q_values.argmax(dim=1).item()
  
  def store_transition(self, state, action, reward, next_state, done):
    self.replay_buffer.push(state, action, reward, next_state, done)

  def update_target_network(self):
    self.target_network.load_state_dict(self.network.state_dict())
  
  def train(self, batch_size=32):
    if len(self.replay_buffer) < batch_size:
      return
    
    batch = self.replay_buffer.sample(batch_size)

    states = torch.FloatTensor(np.array(batch.state)).to(self.device)
    actions = torch.LongTensor(np.array(batch.action)).to(self.device)
    rewards = torch.FloatTensor(np.array(batch.reward)).to(self.device)
    next_states = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
    dones = torch.BoolTensor(np.array(batch.done)).to(self.device)
    
    q_values = self.network(states)
    q_values_taken = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
      next_q_values = self.target_network(next_states)
      max_next_q_values = next_q_values.max(dim=1)[0]
      target_q_values = rewards + self.gamma * max_next_q_values * (~dones).float()

    loss = nn.functional.mse_loss(q_values_taken, target_q_values)
        
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    
    self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    return loss.item()
  
  def save(self, path):
    torch.save(self.network.state_dict(), path)

  def load(self, path):
    self.network.load_state_dict(torch.load(path, map_location=self.device))
    self.target_network.load_state_dict(self.network.state_dict())
