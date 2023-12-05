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
        # Drone state
        state = drone.get_state()
        drone_x, drone_y, _, _, rotation, _ = state
        
        width = abs(min([motor[0] for motor in drone.motors]) - max([motor[0] for motor in drone.motors]))
        height = abs(min([motor[1] for motor in drone.motors]) - max([motor[1] for motor in drone.motors]))

        surface_width = max(width*100 + MOTOR_SIZE, DRONE_SIZE)
        surface_height = max(height*100 + MOTOR_SIZE, DRONE_SIZE)

        drone_surface = pygame.Surface((surface_width, surface_height), pygame.SRCALPHA)  # Use SRCALPHA for transparency

        # Draw the drone rectangle on the surface
        drone_rect = pygame.Rect(surface_width/2 - DRONE_SIZE/2, surface_height/2 - DRONE_SIZE/2, DRONE_SIZE, DRONE_SIZE)
        pygame.draw.rect(drone_surface, WHITE, drone_rect)

        # Draw the motors
        for motor in drone.motors:
            motor_x, motor_y, _ = motor

            motor_x_scaled = motor_x * 100 + surface_width/2
            motor_y_scaled = motor_y * 100 + surface_height/2

            motor_rect = pygame.Rect(motor_x_scaled - MOTOR_SIZE/2, motor_y_scaled - MOTOR_SIZE/2, MOTOR_SIZE, MOTOR_SIZE)
            pygame.draw.rect(drone_surface, RED, motor_rect)

        # Rotate the combined drone and motors surface
        rotation = rotation * 180 / math.pi
        rotated_drone_surface = pygame.transform.rotate(drone_surface, -rotation)

        # Calculate the center position for blitting
        blit_x = drone_x - rotated_drone_surface.get_width() / 2
        blit_y = drone_y - rotated_drone_surface.get_height() / 2

        # Draw the rotated drone surface on the screen
        self.screen.blit(rotated_drone_surface, (blit_x, blit_y))


    def update(self, drone):
       self.clock.tick(60)
       self.screen.fill(BLACK)
       self._draw_drone(drone)
       pygame.display.flip()
