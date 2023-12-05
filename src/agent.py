import torch
import random
import numpy as np
from collections import deque
from src.model import Linear_QNet, QTrainer
from src.drone import Drone

class Agent:
    def __init__(self, config: dict, drone: Drone):
        self.max_x = config['display']['width']
        self.max_y = config['display']['height']
        
        self.n_games = 0
        self.epsilon = config['agent']['epsilon']
        self.epsilon_decay = config['agent']['epsilon_decay']
        self.gamma = config['agent']['gamma']
        self.memory = deque(maxlen=config["agent"]["max_memory"])
        self.batch_size = config["agent"]["batch_size"]

        self.layers = [len(drone.get_state())] + config["agent"]["layers"] + [len(config["drone"]["motors"])]
        print("Layers:", self.layers)
        self.model = Linear_QNet(layers=self.layers, dropout_p=config["agent"]["dropout_p"])
        self.trainer = QTrainer(self.model, lr=config['agent']['learning_rate'], gamma=self.gamma)

    def get_action(self, state: list):
        # random moves: tradeoff exploration / exploitation
        self.epsilon *= self.epsilon_decay

        action = [0] * self.layers[-1]
        # Action is a list of n values (last layer size)
        if random.random() < self.epsilon:
            # List of random values between 0 and 1 of length last layer size
            action = np.random.rand(self.layers[-1])
        else:
            state_tensor = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state_tensor)
            action = prediction.detach().numpy()
    
        return action

    def get_reward(self, state: list, target: dict, done: bool):
        # Negative reward for being out of bounds or other termination criteria
        if done:
            return -10

        # Calculate Euclidean distance from the target
        distance = np.sqrt((target["x"] - state[0]) ** 2 + (target["y"] - state[1]) ** 2)

        distance_reward = 1 / distance
        
        if distance < 5:
            distance_reward = 1

        # Add a constant reward for survival/progress
        constant_reward = 0.1

        #print("Distance Reward:", round(distance_reward,2), "Constant Reward:", round(constant_reward,2))
        return distance_reward + constant_reward

        

    def train_long_memory(self):
        if len(self.memory) > self.batch_size:
            mini_sample = random.sample(self.memory, self.batch_size) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        
    def train_short_memory(self, previous_state, action, reward, state, done):
        self.trainer.train_step(previous_state, action, reward, state, done)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def load(self):
        pass
