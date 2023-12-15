# Libraries
import gymnasium as gym
import numpy as np
import os
from stable_baselines3 import PPO

# Folder to save model and logs
models_dir = "models/PPOCustomized"
logdir = "logsdir"

# Create the folders if they do not exist already
if not os.path.exists(logdir):
    os.makedirs(logdir)

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Create a custom reward wrapper
class CustomRewardWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, done, info, truncated = super().step(action) 
        center_position = [0, 0] # Central position of the landing pad
        reward = self.modify_reward(reward, obs, center_position) # Modify the reward
        return obs, reward, done, info, truncated

    # Modify the reward function as needed
    @staticmethod
    def modify_reward(reward, obs, center_position):
        agent_position = obs[:2] # Get the agent's position
        distance = np.linalg.norm(np.array(agent_position) - np.array(center_position)) # Calculate the distance from the center

        # Get the vertical velocity of the lander
        vertical_velocity = obs[3]

        # Modify the reward to account for distance and vertical velocity
        return reward - distance - 0.1 * np.abs(vertical_velocity)

# Create the Lunar Lander environment
env = CustomRewardWrapper(gym.make('LunarLander-v2'))

# Create the PPO agent
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log = logdir)

# Specify the total number of timesteps and the interval to save the model
total_timesteps = 10000

for i in range(301): # Train the agent for 300 episodes
    model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False, tb_log_name="PPOCustomized") # Train the agent for 10000 timesteps
    model.save(f"{models_dir}/{total_timesteps*i}") # Save the model

# Close the environment
env.close()
