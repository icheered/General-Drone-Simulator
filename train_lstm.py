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

# Read config and set up tensorboard logging
config = read_config("config.yaml")
agent_filename = "best_model"

# Create drone environment with randomization
training_window = 20 # Number of frames per episode to train the LSTM on
env = DroneEnv(config, render_mode=None, max_episode_steps=training_window)
number_training_windows = 250
number_test_windows = 20

agent = PPO.load(os.path.join('training', 'saved_models', agent_filename), env=env, verbose=0)
epochs = 1000

# Create the LSTM model to be trained
hidden_neurons = 250
number_domain_parameters = 3 # Mass, Inertia, Gravity
input_neurons=len(env.get_observation(state=True, domain_params=False, targets=False)) + len(env.motors)
print(f"Input neurons: {input_neurons}")
model = ParameterEstimator(
    sequence_length=training_window, 
    hidden_dim=hidden_neurons, 
    batch_size=number_training_windows, 
    num_lstms=1, 
    input_neurons=input_neurons,
    output_neurons=number_domain_parameters
)
optimizer = optim.Adam(model.parameters())
model.train() # Set the module in training mode

best_test_rmse = -1

# Train the model
for epoch in tqdm(range((epochs))):
    trajectories = []
    labels = []

    # Collect training data
    for i in range(number_training_windows):
        run = []
        obs, _ = env.reset()
        domain_params = env.get_observation(state=False, domain_params=True, targets=False)
        labels.append(domain_params)

        for episode in range(training_window):
            action, _ = agent.predict(obs)
            obs, reward, done, _, info = env.step(action) # Get new set of observations
            state = env.get_observation(state=True, domain_params=False, targets=False)

            lstm_input = np.concatenate((state, action))
            run.append(lstm_input)
        trajectories.append(run)
        run = []

    # Preprocess data
    x_batch, y_batch = model.pre_process(trajectories, labels, training_window, noise_level=0)

    if epoch % 100 == 0:
        # Evaluate the model
        model.eval()
        y_pred = model(x_batch)
        test_rmse = model.RMSE(y_pred, y_batch)
        if best_test_rmse > 0 and test_rmse > best_test_rmse:
            # Save the model as 'best_model'
            save_path = os.path.join('training', 'lstm', 'models')
            savefilename = os.path.join(save_path, "best_model")
            torch.save(model, savefilename)
        print(f"Epoch: {epoch}. Test RMSE: {test_rmse}")
    else: 
        # Train the model
        model.train()
        y_pred = model.forward(x_batch)
        loss = model.custom_loss(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# Save the model and graph to disk
best_model_path = os.path.join('training', 'lstm', 'models', "best_model.zip")
best_model_filename = f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.zip"
os.rename(best_model_path, os.path.join(save_path, best_model_filename+".zip"))
print(f"Training finished. Best model saved as {best_model_filename}")
