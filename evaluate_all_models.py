import os
from stable_baselines3 import PPO
from src.drone_env import DroneEnv
from src.utils import read_config
from tqdm import tqdm
import json


scenarios = {
    "static_static": {
        "title": "Trained: Static. Evaluated: Static",
        "model": "PPO_static_600_02:13:34",
        "environment": {
            "domain_randomization": False,
            "domain_knowledge": False
        }
    },
    "static_dynamic": {
        "title": "Trained: Static. Evaluated: Dynamic",
        "model": "PPO_static_600_02:13:34",
        "environment": {
            "domain_randomization": True,
            "domain_knowledge": False
        }
    },
    "dynamic_static": {
        "title": "Trained: Dynamic. Evaluated: Static",
        "model": "PPO_generalized_318_02:14:08",
        "environment": {
            "domain_randomization": False,
            "domain_knowledge": False
        }
    },
    "dynamic_dynamic": {
        "title": "Trained: Dynamic. Evaluated: Dynamic",
        "model": "PPO_generalized_318_02:14:08",
        "environment": {
            "domain_randomization": True,
            "domain_knowledge": False
        }
    },
    "smart_dynamic_static": {
        "title": "Trained: Dynamic with knowledge. Evaluated: Static",
        "model": "PPO_generalized_with_knowledge_440_02:15:04",
        "environment": {
            "domain_randomization": False,
            "domain_knowledge": True
        }
    },
    "smart_dynamic_dynamic": {
        "title": "Trained: Dynamic with knowledge. Evaluated: Dynamic",
        "model": "PPO_generalized_with_knowledge_440_02:15:04",
        "environment": {
            "domain_randomization": True,
            "domain_knowledge": True
        }
    },

}

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
output_file = os.path.join('results', 'evaluation', 'evaluation.json')
os.makedirs(output_folder, exist_ok=True)  # Create the directory if it doesn't exist

evaluations = load_from_json(output_file)


for name, scenario in scenarios.items():
    config["environment"]["domain_randomization"] = scenario["environment"]["domain_randomization"]
    config["environment"]["domain_knowledge"] = scenario["environment"]["domain_knowledge"]

    env = DroneEnv(config, max_episode_steps=1000)
    model = PPO.load(os.path.join('results', 'saved_models', scenario["model"]), env=env)

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
    
    # Update the evaluations dictionary
    evaluations[name] = {
        **scenario, 
        "evaluation": {
            "rewards": rewards,
            "episode_lengths": episode_lengths
        }
    }

    # Save updated evaluations to JSON
    save_to_json(output_file, evaluations)

    print(f"Updated evaluation for {name} saved to {output_file}")