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
save_path = os.path.join('training', 'saved_models')
log_path = os.path.join('training', 'logs')
logger = configure(log_path, ["stdout", "tensorboard"])

env = DroneEnv(config, render_mode="human", max_episode_steps=1000)
human = Human(len(config["drone"]["motors"]))
while True:
    obs, _ = env.reset()
    done = False
    score = 0 
    
    while not done:
        env.render()
        action = human.get_action()
        obs, reward, done, _, info = env.step(action) # Get new set of observations
        score+=reward
    print('Score:{}'.format(round(score,2)))
