import gymnasium as gym
import ale_py
import cv2
import numpy as np
import torch
import imageio
from collections import deque
from dqn_agent import DQNAgent

# Register the Atari environments
gym.register_envs(ale_py)

def resize_observation(obs):
    obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
    # Return as uint8 (0-255), do NOT divide by 255.0 yet!
    return obs.astype(np.uint8)

def main():
    # Notice render_mode="rgb_array"! This lets us capture the color frames.
    env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
    
    # Initialize the agent
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    agent = DQNAgent(num_actions=env.action_space.n, device=device)
    
    # Load your best checkpoint
    checkpoint_file = "pong_model_ep400.pt" # Change this to your specific checkpoint if needed
    try:
        _, _ = agent.load_checkpoint(checkpoint_file)
        print(f"Successfully loaded {checkpoint_file}!")
    except FileNotFoundError:
        print(f"Error: Could not find {checkpoint_file}. Make sure you've trained the model!")
        return

    # FORCE EXPLORATION TO ZERO. 
    # We want the agent to use 100% brain, 0% random guessing.
    agent.epsilon = 0.00 # Leave a tiny 1% chance to prevent getting stuck in infinite loops
    
    # Setup for a single evaluation episode
    obs, info = env.reset()
    obs = resize_observation(obs)
    
    frame_stack = deque(maxlen=4)
    for _ in range(4):
        frame_stack.append(obs)
        
    stacked = np.array(list(frame_stack))
    
    done = False
    episode_reward = 0
    frames = [] # This will hold all our color images for the GIF

    print("Agent is playing... capturing frames!")
    
    while not done:
        # Capture the high-res color frame for the video
        frames.append(env.render())
        
        # Agent decides what to do
        action = agent.select_action(stacked)
        
        # Step the environment
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_reward += reward
        
        # Update the stack exactly like in training
        obs = resize_observation(observation)
        frame_stack.append(obs)
        stacked = np.array(list(frame_stack))
        
    print(f"Game Over! Agent's Final Score: {episode_reward}")
    env.close()

    # Save the frames as a GIF
    print(f"Saving {len(frames)} frames to 'agent_playing.gif'...")
    # duration=1000/60 means 60 frames per second
    imageio.mimsave('agent_playing.gif', frames, duration=1000/60) 
    print("Done! Open 'agent_playing.gif' to see your AI in action.")

if __name__ == "__main__":
    main()