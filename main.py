import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

def main():
    env = gym.make("ALE/Pong-v5")
    obs, info = env.reset()
    
    print(f"Observation shape: {obs.shape}")
    print(f"Observation dtype: {obs.dtype}")
    print(f"Action space: {env.action_space}")
    print(f"Number of actions: {env.action_space.n}")
    print("\nRunning 50 steps...\n")
    
    done = False
    step_count = 0
    episode_reward = 0
    
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        episode_reward += reward
        step_count += 1
        
        print(f"Step {step_count}: Action={action}, Reward={reward}, Episode Total={episode_reward}")
    
    print(f"\nEpisode ended. Total reward: {episode_reward}")
    env.close()

if __name__ == "__main__":
    main()


