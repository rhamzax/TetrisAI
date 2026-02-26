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
    env = gym.make("ALE/Pong-v5", difficulty=3)
    obs, info = env.reset()

    obs = resize_observation(obs)
    
    frame_stack = deque(maxlen=4)
    for _ in range(4):
        frame_stack.append(obs)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    agent = DQNAgent(num_actions=env.action_space.n, device=device)
    total_steps = 0
    start_episode = 0
    try:
        start_episode, total_steps = agent.load_checkpoint("pong_model_ep50.pt")
        start_episode = 0
        agent.epsilon = 0.50 # Force the agent to explore more at the start of training, since we're starting from a checkpoint
        print("Model loaded from pong_model.pt!")
    except FileNotFoundError:
        print("No saved model found. Starting from scratch.")

    episode_rewards = []
    
    for episode in range(start_episode, 500):  # Train for 500 episodes

        if (episode + 1) % 50 == 0:
            agent.save_checkpoint(f"pong_model_ep{episode+1}.pt", episode, total_steps)
            print(f"Model saved at episode {episode+1}")
            plot_rewards(episode_rewards, window=50)

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
            
            total_steps += 1 # Moved this up so modulo checks align properly
            episode_steps += 1
            episode_reward += reward

            # Train every 4 steps
            if total_steps % 4 == 0:
                agent.train(batch_size=32)
            
            stacked = next_stacked
            
            # Update target network every 10,000 steps
            if total_steps % 10000 == 0:
                agent.update_target_network()
        
        episode_rewards.append(episode_reward)


        print(f"Episode {episode+1}/500: Reward={episode_reward}, Epsilon={agent.epsilon:.3f}, Total Steps={total_steps}")
    
    print(f"Training complete. Average reward (last 10): {np.mean(episode_rewards[-10:]):.2f}")
    agent.save_checkpoint("pong_model.pt", 500, total_steps)
    print("Model saved to pong_model.pt!")
    
    # Generate the graph!
    plot_rewards(episode_rewards, window=50)

    env.close()

def resize_observation(obs):
    obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
    # Return as uint8 (0-255), do NOT divide by 255.0 yet!
    return obs.astype(np.uint8)


def plot_rewards(rewards, window=50):
    """Plots raw rewards and a moving average to visualize training progress."""
    plt.figure(figsize=(10, 5))
    
    # Plot raw noisy rewards in the background
    plt.plot(rewards, label='Raw Episode Reward', alpha=0.3, color='blue')
    
    # Calculate and plot the moving average
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        # Offset the x-axis so the moving average aligns with the end of the window
        plt.plot(range(window - 1, len(rewards)), moving_avg, 
                 label=f'{window}-Episode Moving Avg', color='red', linewidth=2)
    
    plt.title('DQN Pong Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot instead of just showing it, so you don't lose it if you step away
    plt.savefig('pong_training_curve.png', bbox_inches='tight')
    print("Saved training curve plot to 'pong_training_curve.png'!")

if __name__ == "__main__":
    main()