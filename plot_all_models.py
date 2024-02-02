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
# Loop through evaluations and get scenarios with static evaluation_environment.domain_randomization = False
static_scenarios = []
dynamic_scenarios = []

for scenario, data in evaluations.items():
    if not data["evaluation_environment"]["domain_randomization"]:
        static_scenarios.append(scenario)
    else:
        dynamic_scenarios.append(scenario)



# Function to plot and save CDF
def plot_cdf(scenarios, title):
    plt.figure(figsize=(10, 6))
    for scenario in scenarios:
        label = ""
        label += f"{'Dynamic' if evaluations[scenario]['training_environment']['domain_randomization'] else 'Static'}"
        if "domain_estimation" in evaluations[scenario]['training_environment'] and evaluations[scenario]['training_environment']['domain_estimation']:
            label += " with estimation"
        elif evaluations[scenario]['training_environment']['domain_knowledge']:
            label += " with knowledge"
        sns.ecdfplot(evaluations[scenario]["evaluation"]["rewards"], label=label)
    plt.title(title, fontsize=16)
    plt.xlabel('Rewards')
    plt.ylabel('Cumulative Probability')
    plt.xlim(-100, 200)
    plt.legend()

    cdf_title = title.replace(" ", "_") + '_CDF.png'
    output_file = os.path.join('results', 'evaluation', cdf_title)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file)

# Function to plot and save Box Plot
def plot_box(scenarios, title):
    plt.figure(figsize=(10, 6))
    rewards_data = [evaluations[scenario]["evaluation"]["rewards"] for scenario in scenarios]
    sns.boxplot(data=rewards_data)
    labels = []
    for scenario in scenarios:
        label = ""
        label += f"{'Dynamic' if evaluations[scenario]['training_environment']['domain_randomization'] else 'Static'}"
        if "domain_estimation" in evaluations[scenario]['training_environment'] and evaluations[scenario]['training_environment']['domain_estimation']:
            label += " with estimation"
        elif evaluations[scenario]['training_environment']['domain_knowledge']:
            label += " with knowledge"
        labels.append(label)
    plt.xticks(range(len(scenarios)), labels=labels)
    plt.title(title, fontsize=18)
    plt.xlabel('Training Environment', fontsize=14)
    plt.ylabel('Rewards', fontsize=14)

    box_title = title.replace(" ", "_") + '_Box_Plot.png'
    output_file = os.path.join('results', 'evaluation', box_title)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file)

# Plot CDF for static scenarios
plot_cdf(static_scenarios, "Evaluation in Static Environment")

# Plot Box Plot for static scenarios
plot_box(static_scenarios, "Evaluation in Static Environment")

# Plot CDF for dynamic scenarios
plot_cdf(dynamic_scenarios, "Evaluation in Dynamic Environment")

# Plot Box Plot for dynamic scenarios
plot_box(dynamic_scenarios, "Evaluation in Dynamic Environment")