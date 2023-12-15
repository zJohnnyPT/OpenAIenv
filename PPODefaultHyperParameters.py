import gymnasium as gym
import numpy as np
import os
from stable_baselines3 import PPO

models_dir = "models/PPODefaultHyperParameters"
logdir = "logsdir"

if not os.path.exists(logdir):
    os.makedirs(logdir)

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Create the Lunar Lander environment
env = (gym.make('LunarLander-v2'))

# Create the PPO agent
model = PPO(
    "MlpPolicy",
    env,
    n_steps=1024,
    batch_size=64,
    gae_lambda=0.98,
    gamma=0.999,
    n_epochs=4,
    ent_coef=0.01,
    verbose=1,
    tensorboard_log = logdir)

# Specify the total number of timesteps and the interval to save the model
total_timesteps = 10000

iters = 0 
for i in range(301):
    model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False, tb_log_name="PPODefaultHyperParameters")
    model.save(f"{models_dir}/{total_timesteps*i}")

# Close the environment
env.close()
