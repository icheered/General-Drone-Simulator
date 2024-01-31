import datetime
import os
import time

import numpy as np
import torch
from stable_baselines3 import A2C, DQN, HER, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

from src.drone_env import DroneEnv
from src.utils import read_config, format_number
from src.monitor import Monitor
from src.logger_callback import LoggerCallback
from src.reward_callback import StopTrainingOnMovingAverageReward

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: {}".format(device))

# Read config and set up tensorboard logging
config = read_config("config.yaml")
save_path = os.path.join('results', 'saved_models')
figure_path = os.path.join('results', 'figures')
log_path = os.path.join('results', 'logs')
logger = configure(log_path, ["stdout", "tensorboard"])

# Train on a static environment
# Train on a dynamic environment
# Train on a dynamic environment with exact knowledge of domain
scenarios = {
    "static": {
        "domain_randomization": False,
        "domain_knowledge": False
    },
    "generalized": {
        "domain_randomization": True,
        "domain_knowledge": False
    },
    "generalized_with_knowledge": {
        "domain_randomization": True,
        "domain_knowledge": True
    }
}

for name, scenario in scenarios.items():
    print(f"Training on scenario: {name}")

    # Set the config
    config["environment"]["domain_randomization"] = scenario["domain_randomization"]
    config["environment"]["domain_knowledge"] = scenario["domain_knowledge"]
    
    # Set training parameters
    reward_threshold = config["training"]["reward_threshold"]  # Stop training if the average reward is greater or equal to this value
    num_envs = config["training"]["num_envs"]  # Number of parallel environments from which the experience replay buffer is sampled
    max_episode_steps = config["training"]["max_episode_steps"]  # Max number of steps per episode
    max_episodes = config["training"]["episodes"]
    total_timesteps = num_envs * max_episode_steps * max_episodes

    # Create the environment
    model_type = "PPO"
    env_fns = [lambda: DroneEnv(config, render_mode=None, max_episode_steps=max_episode_steps) for _ in range(num_envs)]
    env = DummyVecEnv(env_fns)
    check_env(env.envs[0], warn=True)  # Check if the environment is valid

    #stop_callback = StopTrainingOnRewardThreshold(reward_threshold=reward_threshold, verbose=1)
    eval_callback = EvalCallback(env, 
                                #callback_on_new_best=stop_callback, 
                                eval_freq=1000, 
                                best_model_save_path=save_path, 
                                verbose=1)

    # Monitor handles the plotting of reward and survive time during training
    #monitor = Monitor(config, "PPO")
    #logger_callback = LoggerCallback(monitor=monitor)
    #reward_callback = StopTrainingOnMovingAverageReward(reward_threshold=reward_threshold, window_size=25, verbose=1)
    #callbacks = [eval_callback, logger_callback, reward_callback]
    callbacks = [eval_callback]

    # Create the model
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_path)

    # Do the actual learning
    start_time = time.time()
    try:
        model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=callbacks)
    except KeyboardInterrupt:
        print("Keyboard interrupt detected, exiting training loop")

    # Save the model and graph to disk
    #num_episodes = format_number(len(monitor.rewards))
    training_duration = time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))
    filename = f"{model_type}_{name}_{training_duration}"
    model.save(os.path.join(save_path, filename))
    #monitor.close(os.path.join(figure_path, filename))

    # Copy the 'best_model' to 'filename + _best'
    # best_model_path = os.path.join(save_path, "best_model.zip")
    # best_model_filename = f"{filename}_best"
    # os.rename(best_model_path, os.path.join(save_path, best_model_filename+".zip"))
    print(f"Model saved to {filename}")