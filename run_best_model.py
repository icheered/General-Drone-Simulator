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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))

    # Read config and set up tensorboard logging
    config = read_config("config.yaml")
    filename = "best_model"

    env = DroneEnv(config, render_mode="human", max_episode_steps=1000)
    try:
        while True:
            #model = PPO.load(os.path.join('training', 'saved_models', filename), env=env)
            model = PPO.load(os.path.join('results', 'saved_models', filename), env=env)
            obs, _ = env.reset()
            done = False
            score = 0 
            
            while not done:
                env.render()
                action, _ = model.predict(obs)
                obs, reward, done, _, info = env.step(action) # Get new set of observations
                score+=reward
            print(f'Score: {round(score,2)}')
    except KeyboardInterrupt:
        print("Shutting down...")
    except Exception as e:
        print(e)
    finally:
        env.close()
        exit()

if __name__ == "__main__":
    main()