import datetime
import os

import numpy as np
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: {}".format(device))

# device = torch.device("cpu")
# print("Using device: {}".format(device))

# Read config and set up tensorboard logging
config = read_config("config.yaml")
save_path = os.path.join('training', 'saved_models')
figure_path = os.path.join('training', 'figures')
log_path = os.path.join('training', 'logs')
logger = configure(log_path, ["stdout", "tensorboard"])


# CONTROL THE PROGRAM FLOW
show_env = False
import_last_best_model = False
train_model = True
evaluate_model = False


if show_env:
    # SHOW THE ENVIRONMENT FOR DEBUGGING
    env = DroneEnv(config, render_mode="human", max_episode_steps=1000)
    episodes = 5
    for episode in range(1, episodes+1):
        state = env.reset() # Get initial set of observations
        done = False
        score = 0 
        
        while not done:
            env.render()
            action = env.action_space.sample() # Take a random action from the action space
            n_state, reward, done, _, info =  env.step(action) # Get new set of observations
            score+=reward
        print('Episode:{} Score:{}'.format(episode, round(score,2)))
    env.close()

if train_model:
    # TRAIN THE MODEL
    num_envs = 16  # Number of parallel environments
    reward_threshold = 5000  # Stop training if the mean reward is greater or equal to this value
    max_episode_steps = 1000  # Max number of steps per episode
    total_timesteps = 100000000  # Total number of training steps (ie: environment steps)
    model_type = "PPO"
    # env_fns = [lambda: DroneEnv(config, max_episode_steps=1000) for _ in range(num_envs)]
    env_fns = [lambda: DroneEnv(config, render_mode=None, max_episode_steps=1000) for _ in range(num_envs)]


    # env_fns = []
    # for _ in range(num_envs):
    #     def create_env():
    #         return DroneEnv(config, max_episode_steps=1000)
    #     env_fns.append(create_env)


    env = DummyVecEnv(env_fns)
    check_env(env.envs[0], warn=True)  # Check if the environment is valid

    # Disable rendering for each environment
    for env_instance in env.envs:
        env_instance.enable_rendering = False  # Disable rendering

    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=reward_threshold, verbose=1)
    eval_callback = EvalCallback(env, 
                                callback_on_new_best=stop_callback, 
                                eval_freq=1000, 
                                best_model_save_path=save_path, 
                                verbose=1)

    # Monitor handles the plotting of reward and survive time during training
    monitor = Monitor(config)
    monitor.log_data(1, 1)
    monitor.update_plot()
    logger = LoggerCallback(monitor=monitor)

    callbacks = [eval_callback, logger]

    # Create the model
    model = None
    best_model_path = os.path.join('training', 'saved_models', 'best_model')

   # Create or load the model based on import_last_best_model flag
    if import_last_best_model:
        print("Importing last best model!")
        if model_type == "PPO":
            model = PPO.load(best_model_path, env)
        elif model_type == "A2C":
            model = A2C.load(best_model_path, env)
        elif model_type == "DQN":
            model = DQN.load(best_model_path, env)
        elif model_type == "HER":
            model = HER.load(best_model_path, env)
        else:
            raise ValueError("Model type not specified")
        print("Loaded model from {}".format(best_model_path))
    else:
        if model_type == "PPO":
            model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_path)
        elif model_type == "A2C":
            model = A2C("MlpPolicy", env, verbose=1, tensorboard_log=log_path)
        elif model_type == "DQN":
            model = DQN("MlpPolicy", env, verbose=1, tensorboard_log=log_path)
        elif model_type == "HER":
            model = HER("MlpPolicy", env, verbose=1, tensorboard_log=log_path)
        else:
            raise ValueError("Model type not specified")

    # Do the actual learning
    try:
        model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=callbacks)
    except KeyboardInterrupt:
        print("Keyboard interrupt detected, exiting training loop")
    
    # SAVE THE MODEL TO DISK
    filename = model_type + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model.save(os.path.join(save_path, filename))
    monitor.close(os.path.join(figure_path, filename))
    print("Model saved to {}".format(filename))


if evaluate_model:
    # EVALUATE THE MODEL
    filename = "PPO_Beast_1m"

    env = DroneEnv(config, render_mode="human", max_episode_steps=5000)
    model = PPO.load(os.path.join('training', 'saved_models', filename), env=env)
    evaluate_policy(model, env, n_eval_episodes=5, render=True)
    env.close()