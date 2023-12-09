import os
import datetime 

import torch
from stable_baselines3 import A2C, DQN, HER, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, StopTrainingOnRewardThreshold

from src.drone_env import DroneEnv
from src.utils import read_config
from src.monitor import Monitor
from src.logger_callback import LoggerCallback
from src.human import Human

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: {}".format(device))

# Read config and set up tensorboard logging
config = read_config("config.yaml")

#filename = "PPO_20231209-154908"
filename = "best_model"

env = DroneEnv(config, render_mode="human", max_episode_steps=500)
env = DummyVecEnv([lambda: env])
model = PPO.load(os.path.join('training', 'saved_models', filename), env=env)
evaluate_policy(model, env, n_eval_episodes=5, render=True)
env.close()