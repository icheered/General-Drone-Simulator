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


# Create LSTM model from best model path
save_path = os.path.join('results')
best_model_filename = "LSTM_model.zip"
parameter_estimator = torch.load(os.path.join(save_path, best_model_filename))

# Read config and set up tensorboard logging
config = read_config("config.yaml")
agent_filename = "PPO_generalized_with_knowledge"
training_window = 10    # Number of frames per episode to train the LSTM on

# Create drone environment with randomization
env = DroneEnv(config, render_mode=None, max_episode_steps=training_window)
agent = PPO.load(os.path.join('results', 'saved_models', agent_filename), env=env, verbose=0)
RMSEs = []
mean_RMSEs = []
for i in tqdm(range(100)):
    # Run once and estimate mass and inertia
    obs, _ = env.reset()
    domain_params = env.get_observation(state=False, domain_params=True, targets=False)
    run = []
    rmse = []
    for episode in range(300):
        action, _ = agent.predict(obs)
        obs, reward, done, _, info = env.step(action) # Get new set of observations
        state = env.get_observation(state=True, domain_params=False, targets=False)

        # Include these lines to estimate the domain parameters as input for agent
        lstm_input = np.concatenate((state, action)).tolist()
        run.append(lstm_input)
        if len(run) >= training_window:
            print(f"Shape of run: {np.array(run).shape}")
            x, y = parameter_estimator.pre_process(run, domain_params, training_window)
            parameter_estimator.eval()
            y_pred = parameter_estimator(x)#.detach().numpy() # Probably have to use this as well

            # This is just for testing
            test_rmse = parameter_estimator.RMSE(y_pred, y)
            rmse.append(test_rmse.detach().numpy().tolist())
    mean_RMSEs.append(rmse)
    RMSEs.append(np.mean(rmse))

plt.figure(figsize=(5, 5))
plt.plot(range(300), np.mean(mean_RMSEs, axis=0), marker='o')
plt.show()
print(np.mean(RMSEs))