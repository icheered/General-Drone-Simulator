import torch
import random
import numpy as np
from collections import deque
from model import Linear_QNet, QTrainer
import time

MAX_MEMORY = 100_000
BATCH_SIZE = 100
LR = 0.01  # learning rate

class Agent:
    def __init__(self, layers: list, screen: dict):
        self.max_x = screen['width']
        self.max_y = screen['height']
        
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.99 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()

        self.model = Linear_QNet(layers)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 150 - (self.n_games/25)

        action = [0] * len(self.model.layers[-1])
        # Action is a list of n values (last layer size)
        if random.randint(0, 200) < self.epsilon:
            # List of random values between 0 and 1 of length last layer size
            action = np.random.rand(len(self.model.layers[-1]))
        else:
            action = self.model(state)
    
        return action

    def get_reward(self, state: list):
        # Check if X or Y is out of bounds
        if state[0] <= 0 or state[0] >= self.max_x or state[1] <= 0 or state[1] >= self.max_y:
            return -10
        
        # Check distance from center
        center_x = self.max_x // 2
        center_y = self.max_y // 2
        distance = np.sqrt((center_x - state[0]) ** 2 + (center_y - state[1]) ** 2)

        # Largest reward at center and decreasing reward as distance increases
        # At distance 50 the reward is 0, and at distance 100 the reward is -1
        if(distance < 50):
            return 1
        else:
            return -((distance-50) / 100)
        

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def load(self):
        pass
