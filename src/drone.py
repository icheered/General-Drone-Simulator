import numpy as np
import math
import matplotlib.pyplot as plt
import copy

fig, axes = plt.subplots(3, 1, figsize=(10, 12))
plt.ion()


class Drone:
    def __init__(self, startx: int, starty: int, config: dict, update_frequency: float = 60):
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
        self.update_frequency = update_frequency
        self.dt = 1 / self.update_frequency
        

    def get_state(self):
        return self.state
    
    def print_state(self):
        # Print entire state all rounded to 2 decimals
        print([round(x, 2) for x in self.state])
    
    def plot_state(self):
        # Prepare time steps
        t_steps = [i * self.dt for i in range(len(self.history))]

        # Extract state variables
        x = [state[0] for state in self.history]
        y = [state[1]*-1 for state in self.history]
        vx = [state[2] for state in self.history]
        vy = [state[3] for state in self.history]
        rotation = [state[4] for state in self.history]
        omega = [state[5] for state in self.history]

        # Update plots
        for ax in axes:
            ax.clear()
        
        axes[0].set_title("Position")
        axes[1].set_title("Velocity")
        axes[2].set_title("Rotation and Angular Velocity")

        axes[0].plot(x, y, '.', label='Position (x, y)')
        # Minimal -20 to 20 axis
  

        axes[1].plot(t_steps, vx, '-', label='Velocity X')
        axes[1].plot(t_steps, vy, '-', label='Velocity Y')
        axes[1].set_ylim(-20, 20)
        axes[1].legend()

        axes[2].plot(t_steps, rotation, '-', label='Rotation')
        axes[2].plot(t_steps, omega, '-', label='Angular Velocity')
        axes[2].legend()

        plt.pause(0.0001)
        plt.draw()

        

    def update_state(self, inputs: list):
        # Calculate net force and torque
        net_force = np.array([0.0, 0.0])
        net_torque = 0.0
        for motor, thrust in zip(self.motors, inputs):
            motor_rotation_radians = math.radians(motor[2])
            motor_direction = motor_rotation_radians + self.state[4]

            force_vector = np.array([np.cos(motor_direction) * thrust, np.sin(motor_direction) * thrust])
            net_force += force_vector

            position_vector = np.array([motor[0], motor[1]])
            torque = position_vector[0] * force_vector[1] - position_vector[1] * force_vector[0]
            net_torque += torque

        # Calculate linear acceleration
        linear_acceleration = net_force / self.mass

        # Update linear velocity and position
        self.state[2] += linear_acceleration[0] * self.dt  # Update velocity x
        self.state[3] += linear_acceleration[1] * self.dt  # Update velocity y
        self.state[0] += self.state[2] * self.dt  # Update position x
        self.state[1] += self.state[3] * self.dt  # Update position y

        # Calculate angular acceleration
        angular_acceleration = net_torque / self.inertia

        # Update angular velocity and rotation
        self.state[5] += angular_acceleration * self.dt  # Update angular velocity
        self.state[4] += self.state[5] * self.dt  # Update rotation

        # Save history
        self.history.append(copy.deepcopy(self.state))
        
        if len(self.history) % 50 == 0:
            self.plot_state()
            


        return self.state
        # Do: account for rotation
        # a = F/m
        # v = v_0 + a * t
        # s = s_0 + v * t

        # Do: Account for angles
        # Torque = r x F # r is the distance from the center of mass to the point of force application
        # alpha = torque / I # I is the moment of inertia
        # omega = omega_0 + alpha * t
        # theta = theta_0 + omega * t