import math

class Motor:
    def __init__(self, x: float, y: float, rotation: float, thrust: float = 0.0):
        self.x = x
        self.y = y
        self.rotation = rotation
        self.thrust = 0

    def apply_thrust(self, amount: float):
        self.thrust = amount

class Drone:
    def __init__(self, x: float = 0.0, y: float = 0.0, rotation: float = 0.0, update_frequency: float = 60):
        self.x = x
        self.y = y
        self.velocity_x = 0
        self.velocity_y = 0
        self.rotation = rotation
        self.update_frequency = update_frequency
        self.rotation_velocity = 0
        self.motors = []
        self.history = []
    
    def add_motor(self, motor: Motor):
        self.motors.append(motor)

    def update_position(self):
        time_step = 1 / self.update_frequency
        # Update rotation
        for motor in self.motors:
            rotation_effect = motor.x * motor.thrust
            self.rotation_velocity += rotation_effect * time_step
        self.rotation += self.rotation_velocity * time_step

        # Update velocity
        for motor in self.motors:
            thrust_x = math.cos(math.radians(motor.rotation + self.rotation)) * motor.thrust
            thrust_y = math.sin(math.radians(motor.rotation + self.rotation)) * motor.thrust
            self.velocity_x += thrust_x * time_step * 50
            self.velocity_y += thrust_y * time_step * 50

        # Update position
        self.x += self.velocity_x * time_step 
        self.y -= self.velocity_y * time_step

        if(self.y > 600):
            self.y = 600
            self.velocity_y = 0
        elif(self.y < 0):
            self.y = 0
            self.velocity_y = 0
        
        if(self.x > 800):
            self.x = 800
            self.velocity_x = 0
        elif(self.x < 0):
            self.x = 0
            self.velocity_x = 0
        
        # Save history
        net_thrust = sum(motor.thrust for motor in self.motors)

        self.history.append((self.x, self.y, net_thrust))