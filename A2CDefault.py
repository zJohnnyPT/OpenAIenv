# Libraries
import gymnasium as gym
import numpy as np
import os
from stable_baselines3 import A2C

# Folder to save model and logs
models_dir = "models/A2CDefault"
logdir = "logsdir"

# Create the folders if they do not exist already
if not os.path.exists(logdir):
    os.makedirs(logdir)

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Create the Lunar Lander environment
env = (gym.make('LunarLander-v2'))

# Create the A2C agent
model = A2C('MlpPolicy', env, verbose=1, tensorboard_log = logdir)

# Specify the total number of timesteps and the interval to save the model
total_timesteps = 10000

for i in range(301): # Train the agent for 300 episodes
    model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False, tb_log_name="A2CDefault") # Train the agent for 10000 timesteps
    model.save(f"{models_dir}/{total_timesteps*i}") # Save the model

# Close the environment
env.close()
