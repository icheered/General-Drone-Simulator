import pygame
import sys
import math

from drone import Motor, Drone
from utilities import read_motor_config, colors
from screen import update_screen, draw_motor, draw_drone, screen


THRUST = 1
GRAVITY = 2

def main():
    drone = Drone(400, 300)
    motors = read_motor_config('drone_config.txt')
    for motor in motors:
        drone.add_motor(Motor(motor[0], motor[1], motor[2]))
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4]:
                    motor_index = event.key - pygame.K_1
                    if motor_index < len(motors):
                        drone.motors[motor_index].apply_thrust(THRUST)
            elif event.type == pygame.KEYUP:
                if event.key in [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4]:
                    motor_index = event.key - pygame.K_1  # Assuming keys 1-4 for motors
                    if motor_index < len(motors):
                        drone.motors[motor_index].apply_thrust(0)

        drone.update_position(1/60)
        screen.fill(colors["WHITE"])
        drone.velocity_y -= GRAVITY
        #for motor in drone.motors:
        #    draw_motor(motor.x, motor.y, motor.rotation, motor.thrust > 0)
        draw_drone(drone)
        update_screen()
        print(f"Drone Position: ({drone.x:.1f}, {drone.y:.1f}), Rotation: {drone.rotation:.1f} | Motor 1 rotation: ({drone.motors[1].rotation:.1f}) | Motor 2 rotation: ({drone.motors[1].rotation:.1f})")



    pygame.quit()
    sys.exit()



if __name__ == "__main__":
    main()



# WIDTH, HEIGHT = 800, 600

# drone = Drone(WIDTH//2, HEIGHT//2)


# drone_pos = [WIDTH//2, HEIGHT//2, 0]  # Position: x, y, rotation
# drone_vel = [0, 0, 0]  # Velocity: x, y, rotation
# THRUST = 1.0  # Thrust magnitude
# ROTATE_SPEED = 1.0  # Speed of rotation


# # Game loop
# running = True
# while running:
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             running = False
#         elif event.type == pygame.KEYDOWN:
#             if event.key in [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4]:
#                 motor_index = event.key - pygame.K_1  # Assuming keys 1-4 for motors
#                 if motor_index < len(motors):
#                     motors[motor_index][3] = True  # Activate motor
#         elif event.type == pygame.KEYUP:
#             if event.key in [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4]:
#                 motor_index = event.key - pygame.K_1  # Assuming keys 1-4 for motors
#                 if motor_index < len(motors):
#                     motors[motor_index][3] = False  # Deactivate motor

    

#     # Apply thrust from active motors
#     for motor in motors:
#         if motor[3]:
#             apply_thrust(motor[0], motor[1], motor_orientation)

#     # Rotate the drone based on active motors
#     rotate_drone()
#     move_drone()

#     # Stop at the floor and walls
#     if drone_y >= HEIGHT - motor_size[1]:
#         drone_y = HEIGHT - motor_size[1]
#         velocity_y = 0
#     if drone_x < 0 or drone_x > WIDTH:
#         velocity_x = -velocity_x

#     # Drawing
#     screen.fill(WHITE)
#     # Draw motors and connect them
#     for motor in motors:
#         motor_orientation = (motor[2] + rotation) % 360
#         draw_motor(drone_x + motor[0], drone_y + motor[1], motor_orientation, motor[3])
#     pygame.draw.lines(screen, BLACK, True, [(drone_x + x, drone_y + y) for x, y, _, _ in motors], 3)

#     pygame.display.flip()
#     clock.tick(FPS)

