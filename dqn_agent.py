import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
from dqn_network import DQNNetwork

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity=100000):
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
        
        # Fixed Initialization: Set epsilon start and end values clearly
        self.epsilon = 1.0
        self.epsilon_end = 0.05
        # Removed the exponential decay multiplier since we are using linear decay now

        self.network = DQNNetwork(input_shape=(4, 84, 84), num_actions=num_actions).to(device)
        self.target_network = DQNNetwork(input_shape=(4, 84, 84), num_actions=num_actions).to(device)
        self.target_network.load_state_dict(self.network.state_dict())

        # Lowered learning rate to 1e-4 for stable DQN training
        self.optimizer = optim.Adam(self.network.parameters(), lr=1e-4)
        self.replay_buffer = ReplayBuffer()

    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device) / 255.0  # Normalize here
                q_values = self.network(state_tensor)
                # if np.random.random() < 0.01:
                #     print(f"Q-values: {q_values}, Max: {q_values.max().item():.4f}")
                return q_values.argmax(dim=1).item()
    
    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update_target_network(self):
        self.target_network.load_state_dict(self.network.state_dict())
    
    def train(self, batch_size=32):
        # Added Replay Buffer Warm-up: Wait for 10,000 experiences before training
        if len(self.replay_buffer) < 10000:
            return
        
        batch = self.replay_buffer.sample(batch_size)

        # Convert to float32 and normalize ONLY when creating the tensor batch
        states = torch.FloatTensor(np.array(batch.state)).to(self.device) / 255.0
        next_states = torch.FloatTensor(np.array(batch.next_state)).to(self.device) / 255.0

        actions = torch.LongTensor(np.array(batch.action)).to(self.device)
        rewards = torch.FloatTensor(np.array(batch.reward)).to(self.device)
        dones = torch.BoolTensor(np.array(batch.done)).to(self.device)

        q_values = self.network(states)
        q_values_taken = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            #Vanilla DQN target calculation (for reference):
            # next_q_values = self.target_network(next_states)
            # max_next_q_values = next_q_values.max(dim=1)[0]
            # target_q_values = rewards + self.gamma * max_next_q_values * (~dones).float()

            # Double DQN target calculation:
            # 1. Main network chooses the best action
            best_next_actions = self.network(next_states).argmax(dim=1)
            
            # 2. Target network evaluates that specific action
            next_q_values = self.target_network(next_states)
            evaluated_next_q_values = next_q_values.gather(1, best_next_actions.unsqueeze(1)).squeeze(1)
            
            # 3. Calculate target Q using the evaluated values
            target_q_values = rewards + self.gamma * evaluated_next_q_values * (~dones).float()

        loss = nn.functional.smooth_l1_loss(q_values_taken, target_q_values)
        
        # if np.random.random() < 0.01:
        #     print(f"Loss: {loss.item():.6f}, Q-values range: [{q_values_taken.min():.4f}, {q_values_taken.max():.4f}]")
            
        self.optimizer.zero_grad()
        loss.backward()
        
        # Optional but highly recommended: Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Linear epsilon decay applied safely
        self.epsilon = max(self.epsilon_end, self.epsilon - (1.0 / 100000))
        
        return loss.item()
    
    def save_checkpoint(self, path, episode, total_steps):
        """Saves EVERYTHING so crashes don't matter."""
        checkpoint = {
            'episode': episode,
            'total_steps': total_steps,
            'epsilon': self.epsilon,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        """Restores the exact state of the universe."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        return checkpoint['episode'], checkpoint['total_steps']
    
    def load(self, path):
        self.network.load_state_dict(torch.load(path, map_location=self.device))
        self.target_network.load_state_dict(self.network.state_dict())