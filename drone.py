import math

class Motor:
    def __init__(self, x: float, y: float, rotation: float):
        self.x = x
        self.y = y
        self.rotation = rotation
        self.thrust = 0

    def apply_thrust(self, amount: float):
        self.thrust = amount

class Drone:
    def __init__(self, x: float = 0.0, y: float = 0.0, rotation: float = 0.0):
        self.x = x
        self.y = y
        self.velocity_x = 0
        self.velocity_y = 0
        self.rotation = rotation
        self.rotation_velocity = 0
        self.motors = []
    
    def add_motor(self, motor: Motor):
        self.motors.append(motor)

    def update_position(self, time_step: float):
        # Update rotation
        for motor in self.motors:
            rotation_effect = motor.x * motor.thrust
            self.rotation_velocity += rotation_effect * time_step
        self.rotation += self.rotation_velocity * time_step

        # Update velocity
        for motor in self.motors:
            thrust_x = math.cos(math.radians(motor.rotation + self.rotation)) * motor.thrust * 200
            thrust_y = math.sin(math.radians(motor.rotation + self.rotation)) * motor.thrust * 200
            self.velocity_x += thrust_x * time_step
            self.velocity_y += thrust_y * time_step

        # Update position
        self.x += self.velocity_x * time_step 
        self.y -= self.velocity_y * time_step

        if(self.y > 600):
            self.y = 600
            self.velocity_y = 0
        elif(self.y < 0):
            self.y = 0
        
        if(self.x > 800):
            self.x = 800
            self.velocity_x = 0
        elif(self.x < 0):
            self.x = 0

    def __str__(self):
        return f"Drone Position: ({self.x}, {self.y}), Rotation: {self.rotation}"

def main():
    drone = Drone()
    drone.add_motor(Motor(-100, 0, 90))
    drone.add_motor(Motor(100, 0, 90))
    time_step = 0.1 

    while True:
        print(drone)
        motor_num = int(input("Select motor (1 or 2): "))
        thrust = float(input("Enter thrust amount: "))
        drone.motors[motor_num - 1].apply_thrust(thrust)
        drone.update_position(time_step)

if __name__ == "__main__":
    main()
