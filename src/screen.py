import pygame
import math

from src.utilities import colors

FPS = 60

clock = pygame.time.Clock()

motor_size = (20, 10)

def update_screen():
    pygame.display.flip()
    clock.tick(FPS)

def draw_drone(drone, screen):
    # Determine the bounding box for the motors
    motor_coords_x = [motor.x for motor in drone.motors] + [0]  # Include origin
    motor_coords_y = [motor.y for motor in drone.motors] + [0]  # Include origin
    min_x = min(motor_coords_x)
    max_x = max(motor_coords_x)
    min_y = min(motor_coords_y)
    max_y = max(motor_coords_y)

    # Calculate width and height of the bounding box
    width = max(max_x - min_x, 30) 
    height = max(max_y - min_y, 30)

    # Create a surface for the drone based on the bounding box
    drone_surf = pygame.Surface((width, height), pygame.SRCALPHA)

    # Draw the rectangle (drone) onto this surface
    rect = pygame.Rect(0,0, width, height)
    pygame.draw.rect(drone_surf, colors["BLUE"], rect)

    # Rotate the surface
    rotated_surf = pygame.transform.rotate(drone_surf, drone.rotation)

    # Calculate the center position of the drone
    drone_center_x = drone.x  # The center x coordinate
    drone_center_y = drone.y  # The center y coordinate

    # Calculate the new position for the rotated surface
    rotated_rect = rotated_surf.get_rect(center=(drone_center_x, drone_center_y))
    screen.blit(rotated_surf, rotated_rect.topleft)

    # Draw the motors
    for motor in drone.motors:
        draw_motor(drone, motor, screen)

def draw_motor(drone, motor, screen):
    # Motor size, change as needed
    motor_size = (20, 10)  # example size for the motor

    # Calculate the actual position of the motor
    motor_x, motor_y = rotate_point(motor.x, motor.y, -drone.rotation)
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

    # Correct calculation
    rotated_x = x * cos - y * sin
    rotated_y = x * sin + y * cos
    return rotated_x, rotated_y


def draw_drone_path(drone, screen, config):
    # Assume max_thrust is known or calculate it as shown previously
    max_thrust = config["drone"]["thrust"] * len(drone.motors)

    for i in range(len(drone.history) - 1):
        # Calculate the average thrust between two history points
        avg_thrust = (drone.history[i][2] + drone.history[i+1][2]) / 2
        thrust_percentage = avg_thrust / max_thrust

        # Interpolate between red and green based on the thrust
        color = (
            int((1 - thrust_percentage) * 255),  # Red component decreases with thrust
            int(thrust_percentage * 255),        # Green component increases with thrust
            0                                   # Blue component stays 0
        )

        pygame.draw.line(screen, color, drone.history[i][:2], drone.history[i+1][:2], 3)
