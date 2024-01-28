#
#   The DroneEnv class is used to create a drone environment that can be used with OpenAI Gym.
#   It includes the drone state, kinematic equations, and reward function
#

from gymnasium import Env
from gymnasium.spaces import Box, Discrete
import numpy as np
import random
import math
from src.display import Display
import time

class DroneEnv(Env):
    def __init__(self, config: dict, render_mode=None, max_episode_steps=1000):
        self.motors = config["drone"]["motors"]

        self.mass_config = config["drone"]["mass"]
        self.inertia_config = config["drone"]["inertia"]
        self.gravity_config = config["drone"]["gravity"]

        self.update_frequency = config["display"]["update_frequency"]
        self.dt = 1 / self.update_frequency

        # Number of targets and domain randomization toggle
        self.environment = config["environment"]

        # Every motor activation is between 0 and 1
        self.action_space = Box(np.zeros(len(self.motors)), np.ones(len(self.motors)))
        
        # Observation space is the drone state (x, vx, y, vy, theta, vtheta) and the target positions
        observation_state_range = [
            [-1, -5, -1, -5, -np.pi, -40],
            [1, 5, 1, 5, np.pi, 40]
        ]

        observation_domain_range = [
            [self.mass_config[0], self.inertia_config[0], self.gravity_config[0]],
            [self.mass_config[-1], self.inertia_config[-1], self.gravity_config[-1]]
        ]

        observation_target_range = [
            [-2, -2] * self.environment["num_targets"],
            [2,2] * self.environment["num_targets"]
        ]

        # Only include the domain parameters if domain knowledge is enabled
        if self.environment["domain_knowledge"]:
            self.observation_space = Box(
                low=np.concatenate(observation_state_range[0] + observation_domain_range[0] + observation_target_range[0], axis=None),
                high=np.concatenate(observation_state_range[1] + observation_domain_range[1] + observation_target_range[1], axis=None),
                dtype=np.float32
            )
        else: 
            self.observation_space = Box(
                low=np.concatenate(observation_state_range[0] + observation_target_range[0], axis=None),
                high=np.concatenate(observation_state_range[1] + observation_target_range[1], axis=None),
                dtype=np.float32
            )

        # Flag to control rendering
        self.enable_rendering = render_mode == "human"

        # Initialize the display
        self.render_mode = render_mode
        self.display = None
        if self.render_mode == "human":
            self.display = Display(config=config, title="Drone Simulation")

        # Reset to initialize the state
        self.max_episode_steps = max_episode_steps
        self.episode_step = 0
        self.last_action = [0] * len(self.motors)
        self.episodes_without_target = 0
        self.hit_targets = 0
        self.reset()

    def get_observation(self, state=True, domain_params=True, targets=True):
        observation = []
        if state:
            observation += self.state
        
        if domain_params and self.environment["domain_knowledge"]:
            # Todo: Use estimator instead of factual values
            observation += [self.mass, self.inertia, self.gravity]
        
        if targets:
            # Subtract the target position from the drone position
            current_position = (self.state[0], self.state[2])
            targets = self.targets.copy()
            for i in range(0, len(targets), 2):
                targets[i] -= current_position[0]
                targets[i+1] -= current_position[1]
        
            observation += targets
        
        return observation
    
    def seed(self, seed=None):
        # Set the seed
        random.seed(seed)
        # Return the seed
        return [seed]
    
    def random_position(self, range_val, exclusion = 0):
        # Choose a random sign (positive or negative)
        sign = 1 if random.random() < 0.5 else -1
        # Generate a random value, excluding the specified range around zero
        return random.uniform(exclusion, range_val) * sign

    def reset(self, seed=None):
        self.episode_step = 0

        # Set the domain parameters
        if self.environment["domain_randomization"]:
            self.mass = random.uniform(self.mass_config[0], self.mass_config[-1])
            self.inertia = random.uniform(self.inertia_config[0], self.inertia_config[-1])
            self.gravity = random.uniform(self.gravity_config[0], self.gravity_config[-1])
        else:
            self.mass = self.mass_config[1]
            self.inertia = self.inertia_config[1]
            self.gravity = self.gravity_config[1]
        
        # Define ranges for randomization
        position_range = 0.7
        exclusion_zone = 0.4  # range around zero to exclude
        velocity_range = 0.2
        rotation_range = 1
        angular_velocity_range = 1

        # Randomize the initial state
        self.state = [
            self.random_position(position_range, exclusion_zone),  # Position x
            self.random_position(position_range, exclusion_zone),  # Position y
            random.uniform(-velocity_range, velocity_range),  # Velocity x
            random.uniform(-velocity_range, velocity_range),  # Velocity y
            random.uniform(-rotation_range, rotation_range),  # Rotation
            random.uniform(-angular_velocity_range, angular_velocity_range),  # Angular velocity
        ]

        if not self.environment["randomize_start_state"]:
            self.state = [0] * 6

        # Randomize the target position
        self.targets = [self.random_position(position_range) for _ in range(self.environment["num_targets"] * 2)]
        self.start_position = (self.state[0], self.state[2])
        self.episodes_without_target = 0
        self.last_reward = 0
        self.hit_targets = 0

        # Update display if initialized
        if self.display is not None:
            self.display.reset()
            self.display.update(self)
        
        # Return the observation (state) as a numpy array
        obs = np.array(self.get_observation(), dtype=np.float32)
        info = {}
        return obs, info
        
    def render(self, mode='human'):
        if not self.enable_rendering:
            return
        if self.render_mode == "human":
            self.display.update(self)
            #time.sleep(0.1)  

    def step(self, action):
        # Increment the survive duration
        self.episode_step += 1
        self.last_action = action
        self.episodes_without_target += 1

        # Apply motor inputs
        self._apply_action(action)
        self._apply_gravity()
        self._update_state_timestep()

        done = self._ensure_state_within_boundaries()
        
        if self.episode_step > self.max_episode_steps:
            done = True

        reward = self._get_reward(done)
        info = {"episode_step": self.episode_step} if done else {}
        truncated = False

        # Convert the observation to a numpy array
        obs = np.array(self.get_observation(), dtype=np.float32)

        return obs, reward, done, truncated, info
    
    def _get_reward(self, done: bool):
        if done:
            return -100
        
        # Reward based on distance to closest target
        current_position = (self.state[0], self.state[2])
        reward = 0

        # Bonus for reaching a target
        for i in range(0, len(self.targets), 2):
            min_distance = 0.1
            if np.linalg.norm(np.array(current_position) - np.array(self.targets[i:i+2])) < min_distance:
                reward += 10
                self.hit_targets += 1

                # Logic to update this target's position
                self.targets[i], self.targets[i+1] = self.random_position(0.8), self.random_position(0.8)
                self.episodes_without_target = 0
        
        # Penalty for not reaching a target
        reward -= min((self.episodes_without_target - 150) * 0.0005, 3)

        self.last_reward = reward

        return reward

    # What is type type of action?
    def _apply_action(self, action):
        # Calculate net force and torque
        net_force = np.array([0.0, 0.0])
        net_torque = 0.0

        # Get rotation angle in degrees (currently in radians)
        #rotation_angle = math.degrees(self.state[4])
        rotation_angle = self.state[4]

        for i, motor in enumerate(self.motors):            
            # Calculate thrust
            thrust = action[i] * motor[3]

            # Force components in motor frame
            force_x = thrust * math.cos(math.radians(motor[2]-90))
            force_y = thrust * math.sin(math.radians(motor[2]-90))

            # Rotate the force vector by the drone's rotation angle
            rotated_force_x = force_x * math.cos(rotation_angle) - force_y * math.sin(rotation_angle)
            rotated_force_y = force_x * math.sin(rotation_angle) + force_y * math.cos(rotation_angle)

            # Update net force
            net_force += np.array([rotated_force_x, rotated_force_y])

            # Calculate the torque
            torque = motor[0] * force_y - motor[1] * force_x    
            net_torque += torque

        # Update linear motion
        acceleration = net_force / self.mass
        self.state[1] += acceleration[0] * self.dt * 0.01 # Update velocity x
        self.state[3] += acceleration[1] * self.dt * 0.01 # Update velocity y
        
        # Update rotational motion
        angular_acceleration = net_torque / self.inertia
        self.state[5] += angular_acceleration * self.dt   # Update angular velocity

    def _apply_gravity(self):
        # Apply gravity
        old_state = self.state[3]
        self.state[3] += self.gravity * self.dt
        
    def _ensure_state_within_boundaries(self):
        done = False
        low, high = self.observation_space.low, self.observation_space.high

        for i in range(len(self.state)):
            # Check if the state is out of bounds
            if self.state[i] < low[i] or self.state[i] > high[i]:
                # If the state is out of bounds, adjust it and flag the episode as done
                self.state[i] = np.clip(self.state[i], low[i], high[i])
                done = True

        return done
    
    def _update_state_timestep(self):
        # Update state
        self.state[0] += self.state[1] * self.dt  # Update position x
        self.state[2] += self.state[3] * self.dt  # Update position y
        self.state[4] += self.state[5] * self.dt  # Update rotation
        
        # Ensure the rotation stays within -pi to pi
        self.state[4] = math.atan2(math.sin(self.state[4]), math.cos(self.state[4]))

    def close(self):
        # Call super class
        super().close()
        # Close the display
        if(self.enable_rendering):
            print("Closing display")
            self.display.close()
