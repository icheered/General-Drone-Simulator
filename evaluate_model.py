import os

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.evaluation import evaluate_policy

from src.drone_env import DroneEnv
from src.utils import read_config

config = read_config("config.yaml")
env = DroneEnv(config, render_mode="human", max_episode_steps=600)
# model = DQN.load(os.path.join('training', 'saved_models', 'DQN_model_1M'), env=env)
model = PPO.load(os.path.join('training', 'saved_models', 'STABLE_PPO_model_0.5m'), env=env)

evaluate_policy(model, env, n_eval_episodes=5, render=True)
env.close()