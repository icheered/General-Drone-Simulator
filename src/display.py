import pygame
import math

pygame.init()

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
DRONE_SIZE = 40  # Size of the drone square
MOTOR_SIZE = 10   # Size of the motor squares

class Display:
    def __init__(self, config: dict, update_frequency: int, title: str):
        self.width = config["width"]
        self.height = config["height"]
        self.update_frequency = update_frequency

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()

    def _draw_drone(self, drone):
        # Draw the drone body
        state = drone.get_state()
        drone_x, drone_y, _, _, rotation, _ = state


        drone_rect = pygame.Rect(drone_x - DRONE_SIZE/2, drone_y - DRONE_SIZE/2, DRONE_SIZE, DRONE_SIZE)
        pygame.draw.rect(self.screen, WHITE, drone_rect)

        # Draw the motors
        for motor in drone.motors:
            motor_x, motor_y, _ = motor

            # Rotate motor position around the center of the drone
            rotated_x = (motor_x * math.cos(rotation)) - (motor_y * math.sin(rotation))
            rotated_y = (motor_x * math.sin(rotation)) + (motor_y * math.cos(rotation))

            # Translate to drone's position
            motor_rect = pygame.Rect(drone_x + rotated_x - MOTOR_SIZE/2, drone_y + rotated_y - MOTOR_SIZE/2, MOTOR_SIZE, MOTOR_SIZE)
            pygame.draw.rect(self.screen, RED, motor_rect)


    def update(self, drone):
       self.clock.tick(60)
       self.screen.fill(BLACK)
       self._draw_drone(drone)
       pygame.display.flip()
