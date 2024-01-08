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
        self.mass = config["drone"]["mass"]
        self.inertia = config["drone"]["inertia"]
        self.gravity = config["drone"]["gravity"]

        self.update_frequency = config["display"]["update_frequency"]
        self.dt = 1 / self.update_frequency

        # # Initialize start and target positions
        # self.start_position = (config["start"]["x"], config["start"]["y"])
        # self.target_position = (config["target"]["x"], config["target"]["y"])

        # Action space is 2 motors, each either 0 or 1
        # DQN can only handle discrete action spaces
        # Every action (both motors off, both on, left on, right on) is a discrete value
        self.action_space = Discrete(2 ** len(self.motors))

        # State space is 6 values: x, vx, y, vy, theta, omega
        # x and y are limited between -1, 1
        # vx and vy are velocities and limited between -5, 5
        # theta is rotation and limited between -pi, pi
        # omega is angular velocity and limited between -10, 10
        self.observation_space = Box(
            low=np.array([-2.1, -5, -2.1, -5, -np.pi, -20]),
            high=np.array([2.1, 5, 2.1, 5, np.pi, 20]),
            dtype=np.float32
        )
             # Flag to control rendering
        self.enable_rendering = render_mode == "human"

        

        # Reset to initialize the state 
        self.max_episode_steps = max_episode_steps
        self.episode_step = 0
        self.reset()
        self.last_action = 0            

        # Initialize the display
        # self.start_position = (0,0)
        self.target_position = (0,0)
        self.render_mode = render_mode
        self.display = None
        if(self.render_mode == "human"):
            self.display = Display(config=config, title="Drone Simulation")

    def get_state(self):
        return self.state
    
    def seed(self, seed=None):
        # Set the seed
        random.seed(seed)
        # Return the seed
        return [seed]

    def randomize_position(self, base_position, range=0.2):
        """
        Randomize a position within a given range.

        :param base_position: The base position (x or y coordinate).
        :param variation_range: Maximum variation from the base position.
        :return: Randomized position.
        """
        return base_position + np.random.uniform(-range, range)

    def reset(self, seed=None):
        self.episode_step = 0

        self._update_target_position()
        
        # Reset the state
        self.state = [
            0,        # Starting position x
            0,        # Initial velocity x
            0,        # Starting position y
            0,        # Initial velocity y
            0,        # Initial rotation angle
            0         # Initial angular velocity
        ]
        # self.start_position = (self.randomize_position(0, range=0.5),self.randomize_position(0, range=0.5))
        # self.state[0] = self.start_position[0]
        # self.state[2] = self.start_position[1]
        
        self.start_position = (0,0)
        #print(f"State position: {(self.state[0], self.state[2])}, Start position: {self.start_position}")

        obs = self._get_relative_state()
        info = {}
        return obs, info
    
    def _get_relative_state(self):
        # Use deepcopy
        current_state = self.state.copy()
        current_state[0] -= self.target_position[0]
        current_state[2] -= self.target_position[1]
        return  np.array(current_state, dtype=np.float32)
        


    def render(self, mode='human'):
        if not self.enable_rendering:
            return
        if self.render_mode == "human":
            self.display.update(self)
            #time.sleep(0.1)  

    def step(self, action):
        self.last_action = action
        # Increment the survive duration
        self.episode_step += 1

        # Apply motor inputs
        self._apply_action(action)
        self._apply_gravity()
        self._update_state_timestep()
        #print(f"State: {self.state}")

        done = self._ensure_state_within_boundaries()
        
        if self.episode_step > self.max_episode_steps:
            done = True

        reward = self._get_reward(done)
        # Check if the drone has stabilized
        # if self._has_stabilized():
        #     done = True
        #     reward = (self.max_episode_steps - self.episode_step) * 1.1

        info = {"episode_step": self.episode_step} if done else {}

        truncated = False

        # Convert state to numpy array with dtype float32, if not already done
        obs = self._get_relative_state()

        return obs, reward, done, truncated, info

    def _reached_target_position(self):
        current_position = (self.state[0], self.state[2])
        distance_to_target = np.linalg.norm(np.array(current_position) - np.array(self.target_position))
        #print(f"Distance to target: {distance_to_target}")
        return distance_to_target < 0.2
    
    def _update_target_position(self):
        new_target_x = self.randomize_position(0, range=0.8)
        new_target_y = self.randomize_position(0, range=0.8)
        self.target_position = (new_target_x, new_target_y)
    
    def _get_reward(self, done: bool):
        if done:
            return -100
        if self._reached_target_position():
            self._update_target_position()
            return 100
            
        current_position = (self.state[0], self.state[2])
        distance_to_target = np.linalg.norm(np.array(current_position) - np.array(self.target_position))
        distance_reward = 1.0 / (distance_to_target + 1.0)

        return distance_reward

    # What is type type of action?
    def _apply_action(self, action):
        # Calculate net force and torque
        net_force = np.array([0.0, 0.0])
        net_torque = 0.0

        # Get rotation angle in degrees (currently in radians)
        #rotation_angle = math.degrees(self.state[4])
        rotation_angle = self.state[4]

        # Convert discrete value to list of binary values for each motor
        action = [int(x) for x in list(bin(action)[2:].zfill(len(self.motors)))]
        action.reverse()

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
        self.state[3] += self.gravity * 1 * self.dt
        
    
    def _ensure_state_within_boundaries(self):
        done = False
        low, high = self.observation_space.low, self.observation_space.high

        # Iterate through each element in the state
        for i in range(len(self.state)):
            # Check for lower boundary
            if self.state[i] < low[i]:
                self.state[i] = low[i]
                # Reset velocity to 0 if position is out of bounds
                if i % 2 == 0:  # Assuming even indices are positions and odd indices are velocities
                    self.state[i + 1] = 0
                print(f"State {i} is out of bounds (too low). Is currently: {self.state[i]}, should be min: {low[i]}")
                done = True
            # Check for upper boundary
            elif self.state[i] > high[i]:
                self.state[i] = high[i]
                # Reset velocity to 0 if position is out of bounds
                if i % 2 == 0:  # Assuming even indices are positions and odd indices are velocities
                    self.state[i + 1] = 0
                print(f"State {i} is out of bounds (too high). Is currently: {self.state[i]}, should be min: {high[i]}")
                done = True
        
        # Force it to die
        if self.state[0] > 1:
            done = True
        if self.state[0] < -1:
            done = True
        if self.state[2] > 1:
            done = True
        if self.state[2] < -1:
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
        if(self.render_mode == "human"):
            self.display.close()
