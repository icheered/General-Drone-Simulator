import numpy as np
import math
import matplotlib.pyplot as plt
import copy

fig, axes = plt.subplots(3, 1, figsize=(10, 12))
plt.ion()


class Drone:
    def __init__(self, startx: int, starty: int, display: dict, config: dict, update_frequency: float = 60):
        # Positions global, rotations local
        self.state = [
            startx,  # Position x
            starty,  # Position y
            0,  # Velocity x
            0,  # Velocity y
            0,  # Rotation
            0,  # Angular velocity
        ]
        self.history = [self.state]

        self.motors = config["motors"]
        self.mass = config["mass"]
        self.inertia = config["inertia"]
        self.thrust = config["thrust"]
        self.gravity = config["gravity"]

        self.max_x = display["width"]
        self.max_y = display["height"]

        self.update_frequency = update_frequency
        self.dt = 1 / self.update_frequency
        

    def get_state(self):
        return self.state
    
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
        # Ensure drone stays within the screen
        if self.state[0] < 0:
            self.state[0] = 0
            self.state[2] = 0
        elif self.state[0] > self.max_x:
            self.state[0] = self.max_x
            self.state[2] = 0

        if self.state[1] < 0:
            self.state[1] = 0
            self.state[3] = 0
        elif self.state[1] > self.max_y:
            self.state[1] = self.max_y
            self.state[3] = 0
    
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
        self._ensure_drone_within_screen()

        return self.state    
