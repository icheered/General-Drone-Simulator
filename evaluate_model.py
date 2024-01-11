import os
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from src.drone_env import DroneEnv
from src.utils import read_config

config = read_config("config.yaml")
env = DroneEnv(config, max_episode_steps=1000)
model = PPO.load(os.path.join('training', 'saved_models', 'best_model'), env=env)

# Evaluate the policy
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)

# Print the results
print(f"Mean Reward: {mean_reward}, Std Reward: {std_reward}")

env.close()
