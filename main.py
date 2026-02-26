import gymnasium as gym
import ale_py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import torch

from dqn_network import DQNNetwork
from dqn_agent import DQNAgent

gym.register_envs(ale_py)


def main():
    env = gym.make("ALE/Pong-v5")
    obs, info = env.reset()

    obs = resize_observation(obs)
    
    frame_stack = deque(maxlen=4)
    for _ in range(4):
        frame_stack.append(obs)

    device = torch.device("cpu")
    agent = DQNAgent(num_actions=env.action_space.n, device=device)

    try:
        agent.load("pong_model.pt")
        print("Model loaded from pong_model.pt!")
    except FileNotFoundError:
        print("No saved model found. Starting from scratch.")

    total_steps = 0
    episode_rewards = []
    
    for episode in range(100):  # Train for 100 episodes
        obs, info = env.reset()
        obs = resize_observation(obs)
        
        frame_stack = deque(maxlen=4)
        for _ in range(4):
            frame_stack.append(obs)
        
        stacked = np.array(list(frame_stack))
        done = False
        episode_reward = 0
        episode_steps = 0
        
        while not done:
            action = agent.select_action(stacked)
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            obs = resize_observation(observation)
            frame_stack.append(obs)
            next_stacked = np.array(list(frame_stack))
            
            agent.store_transition(stacked, action, reward, next_stacked, done)
            agent.train(batch_size=32)
            
            stacked = next_stacked
            episode_reward += reward
            episode_steps += 1
            total_steps += 1
            
            if total_steps % 10000 == 0:
                agent.update_target_network()
        
        episode_rewards.append(episode_reward)
        print(f"Episode {episode+1}/100: Reward={episode_reward}, Epsilon={agent.epsilon:.3f}, Total Steps={total_steps}")
    
    print(f"Training complete. Average reward (last 10): {np.mean(episode_rewards[-10:]):.2f}")
    agent.save("pong_model.pt")
    print("Model saved to pong_model.pt!")
    env.close()

def resize_observation(obs):
    obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
    obs = obs.astype(np.float32) / 255.0
    return obs

if __name__ == "__main__":
    main()