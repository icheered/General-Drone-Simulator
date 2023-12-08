import os

import numpy as np
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_checker import check_env

from src.drone_env import DroneEnv
from src.utils import read_config

# Create log directory
log_dir = "logs/"
os.makedirs(log_dir, exist_ok=True)

# Configure logger to use TensorBoard
logger = configure(log_dir, ["stdout", "tensorboard"])

config = read_config("config.yaml")

def print_state(state):
    # Print 4 decimals, and a space if the number is positive for alignment
    print("State: ", [f"{x:.4f}" if x < 0 else f"{x:.4f} " for x in state])

# env = DroneEnv(config, render_mode="human")
# episodes = 5
# for episode in range(1, episodes+1):
#     state = env.reset() # Get initial set of observations
#     done = False
#     score = 0 
    
#     while not done:
#         env.render()
#         action = env.action_space.sample() # Take a random action from the action space
#         n_state, reward, done, info =  env.step(action) # Get new set of observations
#         score+=reward
#     print('Episode:{} Score:{}'.format(episode, score))
# env.close()






env = DroneEnv(config)
check_env(env, warn=True)
env = DummyVecEnv([lambda: env])
model = DQN('MlpPolicy', env, tensorboard_log=log_dir)

model.learn(total_timesteps=100000, progress_bar=True)