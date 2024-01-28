import os
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from src.drone_env import DroneEnv
from src.utils import read_config
from tqdm import tqdm

config = read_config("config.yaml")
env = DroneEnv(config, max_episode_steps=1000)

filename = "PPO_331_00-51-10_20240128-221022_best"
model = PPO.load(os.path.join('training', 'saved_models', filename), env=env)

# Manual evaluation of the policy with a progress bar
n_eval_episodes = 1000
rewards = []
episode_lengths = []

for _ in tqdm(range(n_eval_episodes), desc="Evaluating"):
    obs, _ = env.reset()
    done = False
    episode_reward = 0
    episode_length = 0
    
    while not done:
        env.render()
        action, _ = model.predict(obs)
        obs, reward, done, _, info = env.step(action) # Get new set of observations
        episode_reward += reward
        episode_length += 1
    rewards.append(episode_reward)
    episode_lengths.append(episode_length)

env.close()




# Create a histogram for rewards and episode lengths
fig, ax = plt.subplots(1, 2)
ax[0].hist(rewards, bins=50)
ax[0].set_xlabel('Reward')
ax[0].set_ylabel('Frequency')
ax[1].hist(episode_lengths, bins=50)
ax[1].set_xlabel('Episode length')
ax[1].set_ylabel('Frequency')
plt.show()

env.close()
