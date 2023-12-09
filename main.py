import os
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on device: {}".format(device))

import numpy as np
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, StopTrainingOnRewardThreshold

from src.drone_env import DroneEnv
from src.utils import read_config
from src.monitor import Monitor
from src.logger_callback import LoggerCallback

# Create log directory
log_dir = "logs/"
os.makedirs(log_dir, exist_ok=True)

# Configure logger to use TensorBoard
logger = configure(log_dir, ["stdout", "tensorboard"])

config = read_config("config.yaml")

save_path = os.path.join('test_training', 'saved_models')
log_path = os.path.join('test_training', 'logs')

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



num_envs = 8  # Number of parallel environments
env_fns = [lambda: DroneEnv(config, max_episode_steps=1000) for _ in range(num_envs)]
env = DummyVecEnv(env_fns)

check_env(env.envs[0], warn=True)  # Checking only the first instance for compatibility

stop_callback = StopTrainingOnRewardThreshold(reward_threshold=800, verbose=1)
eval_callback = EvalCallback(env, 
                             callback_on_new_best=stop_callback, 
                             eval_freq=1000, 
                             best_model_save_path=save_path, 
                             verbose=1)

monitor = Monitor(config)
monitor.update_plot()
logger = LoggerCallback(monitor=monitor)

callbacks = [eval_callback, logger]

model = PPO('MlpPolicy', env, tensorboard_log=log_dir)
model.learn(total_timesteps=1000000, progress_bar=True, callback=callbacks)


model.save(os.path.join('training', 'saved_models', 'PPO_model_1m'))
# env = DroneEnv(config)
# check_env(env, warn=True)
# env = DummyVecEnv([lambda: env])

# stop_callback = StopTrainingOnRewardThreshold(reward_threshold=1000, verbose=1)
# eval_callback = EvalCallback(env, 
#                              callback_on_new_best=stop_callback, 
#                              eval_freq=10000, 
#                              best_model_save_path=save_path, 
#                              verbose=1)

# monitor = Monitor(config)
# monitor.update_plot()
# logger = LoggerCallback(monitor=monitor)

# callbacks = [eval_callback, logger]

# model = DQN('MlpPolicy', env, tensorboard_log=log_dir)

# model.learn(total_timesteps=100000, progress_bar=True, callback=callbacks)


