import numpy as np
import math
import matplotlib.pyplot as plt
import copy
import random 

fig, axes = plt.subplots(3, 1, figsize=(10, 12))
plt.ion()


class Drone:
    def __init__(self, startx: int, starty: int, display: dict, config: dict, update_frequency: float = 60):
        # Positions global, rotations local
        self.startx = startx
        self.starty = starty
        self.survive_duration = 0
        self.reset_state()

        self.motors = config["motors"]
        self.mass = config["mass"]
        self.inertia = config["inertia"]
        self.thrust = config["thrust"]
        self.gravity = config["gravity"]

        self.max_x = display["width"]
        self.max_y = display["height"]

        self.update_frequency = update_frequency
        self.dt = 1 / self.update_frequency
        

    def reset_state(self):
        self.survive_duration = 0

        # Define ranges for randomization
        position_range = 20  # Adjust as needed
        velocity_range = 10  # Adjust as needed
        rotation_range = 0.1     # Radians, adjust as needed
        angular_velocity_range = 0.8  # Adjust as needed

        self.state = [
            self.startx + random.uniform(-position_range, position_range),  # Position x
            self.starty + random.uniform(-position_range, position_range),  # Position y
            random.uniform(-velocity_range, velocity_range),  # Velocity x
            random.uniform(-velocity_range, velocity_range),  # Velocity y
            random.uniform(-rotation_range, rotation_range),  # Rotation
            random.uniform(-angular_velocity_range, angular_velocity_range),  # Angular velocity
        ]

    def get_state(self):
        return self.state
    
    def get_normalized_state(self, target: dict):
        # Normalize the state to be between 0 and 1
        n_state = copy.deepcopy(self.state)
        n_state[0] = (n_state[0] - target["x"]) / self.max_x
        n_state[1] = (n_state[1] - target["y"]) / self.max_y
        n_state[2] /= self.max_x
        n_state[3] /= self.max_y
        return n_state
    

    def print_state(self):
        # Print entire state all rounded to 2 decimals
        print([round(x, 2) for x in self.state])

    def _apply_action(self, action: list):
        # Calculate net force and torque
        net_force = np.array([0.0, 0.0])
        net_torque = 0.0
        rotation_angle = self.state[4]

        for i, motor in enumerate(self.motors):
            # Calculate thrust
            thrust = action[i] * self.thrust

            # Force components in motor frame
            force_x = thrust * math.cos(math.radians(motor[2]))
            force_y = thrust * math.sin(math.radians(motor[2]))

            # Rotate the force vector by the drone's rotation angle
            rotated_force_x = force_x * math.cos(rotation_angle) - force_y * math.sin(rotation_angle)
            rotated_force_y = force_x * math.sin(rotation_angle) + force_y * math.cos(rotation_angle)

            # Update net force
            net_force += np.array([rotated_force_x, rotated_force_y])

            # Calculate the torque
            torque = motor[0] * force_y  # Only y-component of force contributes to torque
            net_torque += torque

        # Update linear motion
        acceleration = net_force / self.mass
        self.state[2] += acceleration[0] * self.dt  # Update velocity x
        self.state[3] += acceleration[1] * self.dt  # Update velocity y
        
        # Update rotational motion
        angular_acceleration = net_torque / self.inertia
        self.state[5] += angular_acceleration * self.dt  # Update angular velocity
        
    def _apply_gravity(self):
        # Apply gravity
        self.state[3] += self.gravity * 100 * self.dt
    
    def _ensure_drone_within_screen(self):
        done = False
        # Ensure drone stays within the screen
        if self.state[0] < 0:
            self.state[0] = 0
            self.state[2] = 0
            done = True
        elif self.state[0] > self.max_x:
            self.state[0] = self.max_x
            self.state[2] = 0
            done = True

        if self.state[1] < 0:
            self.state[1] = 0
            self.state[3] = 0
            done = True
        elif self.state[1] > self.max_y:
            self.state[1] = self.max_y
            self.state[3] = 0
            done = True

        return done
    
    def _update_state_timestep(self):
        # Update state
        self.state[0] += self.state[2] * self.dt  # Update position x
        self.state[1] += self.state[3] * self.dt  # Update position y
        self.state[4] += self.state[5] * self.dt  # Update rotation
        
        # Ensure the rotation stays within -180 to 180 degrees
        self.state[4] = (self.state[4] + 180) % 360 - 180

    def update_state(self, inputs: list):
        # Apply motor inputs
        self._apply_action(inputs)
        self._apply_gravity()
        self._update_state_timestep()
        done = self._ensure_drone_within_screen()

        self.survive_duration += self.dt

        return done
