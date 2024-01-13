import os

import torch
from stable_baselines3 import PPO
import csv
from src.drone_env import DroneEnv
from src.utils import read_config
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: {}".format(device))

# Read config and set up tensorboard logging
config = read_config("config.yaml")
filename = "best_model"

# Create drone environment with randomization
domain_rand = True
env = DroneEnv(config, render_mode=None, max_episode_steps=1000, domain_rand=domain_rand)
trajectories = [] 
iter = 0
trajectories_len = 5000 # Number of trajectories to generate
try:
    while iter < trajectories_len:
        model = PPO.load(os.path.join('training', 'saved_models', filename), env=env, verbose=0)
        obs, _ = env.reset()
        done = False
        score = 0
        traj_states = []
        traj_actions = []
        traj = []
        
        while not done:
            env.render()
            action, _ = model.predict(obs)
            traj_states.append(obs)
            traj_actions.append(action)
            traj.append(np.concatenate((obs,action)))
            obs, reward, done, _, info = env.step(action) # Get new set of observations
            score+=reward
        
        traj = np.transpose(np.array(traj))
        print("Trajectory has size: [" + str(traj[:,0].size) + ", " + str(traj[0,:].size) + "]")
        
        # Ensure trajectory has at least lenght of 10
        if  traj[0,:].size > 10:
            # Specify the file paths
            csv_file_path1 = "trajectories.csv"
            csv_file_path2 = "labels.csv"

            # Open the CSV files in append mode
            with open(csv_file_path1, 'a', newline='') as csvfile:
                # Create a CSV writer object
                csv_writer = csv.writer(csvfile)
                csv_writer.writerows(traj)
            print(f'Trajectories saved to {csv_file_path1}')
            with open(csv_file_path2, 'a', newline='') as csvfile:
                # Create a CSV writer object
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(env.get_params())
            print(f'Mass saved to {csv_file_path2}')
            
            iter += 1
            
except KeyboardInterrupt:
    print("Shutting down...")
finally:
    env.close()
    exit()