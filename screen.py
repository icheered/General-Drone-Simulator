import pygame
import sys
import math

from utilities import colors

BLACK = (0, 0, 0)
BLUE = (100, 100, 255)
RED = (255, 0, 0)
WHITE = (255, 255, 255)
FPS = 60

pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Drone Simulation")
clock = pygame.time.Clock()

motor_size = (20, 10)

def update_screen():
    pygame.display.flip()
    clock.tick(FPS)

def draw_drone(drone):
    # Create a surface for the drone
    drone_surf = pygame.Surface(motor_size, pygame.SRCALPHA)

    # Draw the rectangle (drone) onto this surface
    rect = pygame.Rect(0, 0, *motor_size)
    pygame.draw.rect(drone_surf, BLACK, rect)

    # Rotate the surface
    rotated_surf = pygame.transform.rotate(drone_surf, drone.rotation)

    # Calculate the new position for the rotated surface
    rotated_rect = rotated_surf.get_rect(center=(drone.x, drone.y))
    screen.blit(rotated_surf, rotated_rect.topleft)

    # Draw the motors
    for motor in drone.motors:
        draw_motor(drone, motor)

def draw_motor(drone, motor):
    # Motor size, change as needed
    motor_size = (20, 10)  # example size for the motor

    # Calculate the actual position of the motor
    motor_x, motor_y = rotate_point(motor.x, motor.y, drone.rotation)
    motor_x += drone.x
    motor_y += drone.y

    # Create a surface for the motor
    motor_surf = pygame.Surface(motor_size, pygame.SRCALPHA)

    # Draw the rectangle (motor) onto this surface
    rect = pygame.Rect(0, 0, *motor_size)
    color = colors["GREEN"] if motor.thrust else colors["RED"]
    pygame.draw.rect(motor_surf, color, rect)  # Draw motor in red for visibility

    # Rotate the surface
    # Add drone's rotation to the motor's rotation
    total_rotation = motor.rotation + drone.rotation
    rotated_surf = pygame.transform.rotate(motor_surf, total_rotation)

    # Calculate the new position for the rotated surface
    rotated_rect = rotated_surf.get_rect(center=(motor_x, motor_y))

    # Blit the rotated surface onto the screen
    screen.blit(rotated_surf, rotated_rect.topleft)


def rotate_point(x, y, angle):
    """Rotate a point around the origin (0, 0) by an angle in degrees."""
    radians = math.radians(angle)
    cos = math.cos(radians)
    sin = math.sin(radians)
    return -x * cos - y * sin, x * sin + y * cos