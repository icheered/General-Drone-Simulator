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
    def __init__(self, config: dict, render_mode = None, max_episode_steps = 1000):
        self.motors = config["drone"]["motors"]
        self.mass = config["drone"]["mass"]
        self.inertia = config["drone"]["inertia"]
        self.gravity = config["drone"]["gravity"]

        self.update_frequency = config["display"]["update_frequency"]
        self.dt = 1 / self.update_frequency

        self.target = {
            "x": config["target"]["x"],
            "y": config["target"]["y"],
            "distance": config["target"]["distance"]
        }

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
            low=np.array([-1, -5, -1, -5, -np.pi, -20]),
            high=np.array([1, 5, 1, 5, np.pi, 20]),
            dtype=np.float32
        )

        # Reset to initialize the state 
        self.max_episode_steps = max_episode_steps
        self.episode_step = 0
        self.reset()
        self.last_action = 0
        
        # Initialize the display
        self.render_mode = render_mode
        if(self.render_mode == "human"):
            self.display = Display(
                config=config,
                title="Drone Simulation"
            )

    def get_state(self):
        return self.state
    
    def seed(self, seed=None):
        # Set the seed
        random.seed(seed)
        # Return the seed
        return [seed]

    def reset(self, seed=None):
        # Reset the survive duration
        self.episode_step = 0

        # Define ranges for randomization
        position_range = 0.7
        exclusion_zone = 0.4  # range around zero to exclude

        velocity_range = 0.4
        rotation_range = 2
        angular_velocity_range = 1

        def random_position(range_val, exclusion):
            # Choose a random sign (positive or negative)
            sign = 1 if random.random() < 0.5 else -1
            # Generate a random value, excluding the specified range around zero
            return random.uniform(exclusion, range_val) * sign

        self.state = [
            random_position(position_range, exclusion_zone),  # Position x
            random_position(position_range, exclusion_zone),  # Position y
            random.uniform(-velocity_range, velocity_range),  # Velocity x
            random.uniform(-velocity_range, velocity_range),  # Velocity y
            random.uniform(-rotation_range, rotation_range),  # Rotation
            random.uniform(-angular_velocity_range, angular_velocity_range),  # Angular velocity
        ]

            
        # self.state = [
        #     0,  # Position x
        #     0,  # Velocity x
        #     0,  # Position y
        #     0,  # Velocity y
        #     0,  # Rotation
        #     0,  # Angular velocity
        # ]

        # Convert the state to a numpy array with dtype float32
        obs = np.array(self.state, dtype=np.float32)
        info = {}
        return obs, info

            

    
    def render(self):
        mode = self.render_mode
        assert mode in ["human", None], "Invalid mode, must be either \"human\" or None"
        if mode == None:
            return
        elif mode == "human":
            self.display.update(self)
            #time.sleep(0.1)


    # What is type type of action?
    def step(self, action):
        self.last_action = action
        # Increment the survive duration
        self.episode_step += 1

        # Apply motor inputs
        self._apply_action(action)
        self._apply_gravity()
        self._update_state_timestep()

        done = self._ensure_state_within_boundaries()
        reward = self._get_reward(done)
        
        if self.episode_step > self.max_episode_steps:
            done = True

        # Check if the drone has stabilized
        if self._has_stabilized():
            done = True
            reward = (self.max_episode_steps - self.episode_step) * 1.1

        info = {"episode_step": self.episode_step} if done else {}

        truncated = False

        # Convert state to numpy array with dtype float32, if not already done
        obs = np.array(self.state, dtype=np.float32)

        return obs, reward, done, truncated, info

    def _has_stabilized(self):
        # If the drone is stable, no need to run the rest of the simulation

        # Check if x and y potision and velocities are below a threshold
        position_threshold = 0.05
        if abs(self.state[0]) > position_threshold or abs(self.state[2]) > position_threshold:
            return False
        
        # Check if linear velocity is below a threshold
        velocity_threshold = 0.05
        if abs(self.state[1]) > velocity_threshold or abs(self.state[3]) > velocity_threshold:
            return False
        
        # Check if rotation is within a threshold of 0
        rotation_threshold = 0.1
        if abs(self.state[4]) > rotation_threshold:
            return False
        
        # Check if angular velocity is below a threshold
        angular_velocity_threshold = 0.2
        if abs(self.state[5]) > angular_velocity_threshold:
            return False
        
        return True
    
    
    def _get_reward(self, done: bool):
        # Calculate reward
        if done:
            return -100
        
        # Calculate Euclidean distance from the target
        distance = np.sqrt(self.state[0] ** 2 + self.state[2] ** 2)

        distance_reward = 1 - distance / self.target["distance"]

        constant_reward = 0.1

        return min(distance_reward + constant_reward, 1.1)


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
                done = True
            # Check for upper boundary
            elif self.state[i] > high[i]:
                self.state[i] = high[i]
                # Reset velocity to 0 if position is out of bounds
                if i % 2 == 0:  # Assuming even indices are positions and odd indices are velocities
                    self.state[i + 1] = 0
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
