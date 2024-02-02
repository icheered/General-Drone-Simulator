import datetime
import os

import torch
import torch.optim as optim
from stable_baselines3 import PPO
from src.drone_env import DroneEnv
from src.utils import read_config
import numpy as np
from tqdm import tqdm
from src.lstm import ParameterEstimator 
import matplotlib.pyplot as plt
import json


# best_model_path = os.path.join('training', 'lstm', 'models', "best_model.zip")
best_model_filename = "LSTM_model.zip"
parameter_estimator = torch.load(os.path.join('results', best_model_filename))

# Read config and set up tensorboard logging
config = read_config("config.yaml")
agent_filename = "PPO_generalized_with_knowledge"
training_window = 10    # Number of frames per episode to train the LSTM on

# Create drone environment with randomization
env = DroneEnv(config, render_mode=None, max_episode_steps=training_window)
agent = PPO.load(os.path.join('results', 'saved_models', agent_filename), env=env, verbose=0)
RMSEs = []
mean_RMSEs = []
episodes = 100
for i in tqdm(range(episodes)):
    # Run once and estimate mass and inertia
    obs, _ = env.reset()
    domain_params = env.get_observation(state=False, domain_params=True, targets=False)
    run = []
    rmse = []
    timesteps = 300
    for episode in range(timesteps):
        action, _ = agent.predict(obs)
        obs, reward, done, _, info = env.step(action) # Get new set of observations
        state = env.get_observation(state=True, domain_params=False, targets=False)

        # Include these lines to estimate the domain parameters as input for agent
        lstm_input = np.concatenate((state, action)).tolist()
        run.append(lstm_input)
        #if episode >= training_window:
        x, y = parameter_estimator.pre_process(run, domain_params, training_window)
        parameter_estimator.eval()
        y_pred = parameter_estimator(x)#.detach().numpy() # Probably have to use this as well

        # This is just for testing
        test_rmse = parameter_estimator.RMSE(y_pred, y)
        rmse.append(test_rmse.detach().numpy().tolist())
    mean_RMSEs.append(rmse)
    RMSEs.append(np.mean(rmse))

# Save the RMSEs to a file
now = datetime.datetime.now()
output_folder = os.path.join('results', 'evaluation')
output_file = os.path.join(output_folder, 'RMSEs.json')
os.makedirs(output_folder, exist_ok=True)  # Create the directory if it doesn't exist
with open(output_file, 'w') as file:
    json.dump(RMSEs, file, indent=4)




plt.figure(figsize=(5, 5))
plt.plot(range(300), np.mean(mean_RMSEs, axis=0))
plt.xlabel('Episode')
plt.ylabel('RMSE')
plt.title(f'Average RMSE of {episodes} runs of {timesteps} steps')
# Save
output_file = os.path.join('results', 'evaluation', 'lstm.png')
os.makedirs(os.path.dirname(output_file), exist_ok=True)
plt.savefig(output_file)