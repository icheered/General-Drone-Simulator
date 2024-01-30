import os
from stable_baselines3 import PPO
from src.drone_env import DroneEnv
from src.utils import read_config
from tqdm import tqdm
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

matplotlib.use('Agg')  # Use a non-interactive backend such as Agg

def save_to_json(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def load_from_json(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)
    else:
        return {}
    
config = read_config("config.yaml")
output_folder = os.path.join('results', 'evaluation')
input_file = os.path.join('results', 'evaluation', 'evaluation.json')
os.makedirs(output_folder, exist_ok=True)  # Create the directory if it doesn't exist

evaluations = load_from_json(input_file)

# Create 2 plots. 1 for rewards for scenarios with static evaluation environment and 1 for dynamic evaluation environment
static_scenarios = ["static_static", "dynamic_static", "smart_dynamic_static"]
dynamic_scenarios = ["static_dynamic", "dynamic_dynamic", "smart_dynamic_dynamic"]

# Create a plot for each scenario
def plot_kde(scenarios, title):
    plt.figure(figsize=(10, 6))
    for scenario in scenarios:
        sns.kdeplot(evaluations[scenario]["evaluation"]["rewards"], label=scenario)
    plt.title(title)
    plt.xlabel('Rewards')
    plt.ylabel('Density')
    plt.xlim(-100, 200)
    plt.legend()

    # Save plot to file
    # Remove spaces and make camel case
    title = title.replace(" ", "_")
    output_file = os.path.join('results', 'evaluation', title + '.png')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file)

# Plot for static scenarios
plot_kde(static_scenarios, "Evaluation in Static Environment")

# Plot for dynamic scenarios
plot_kde(dynamic_scenarios, "Evaluation in Dynamic Environment")