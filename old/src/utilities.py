import pygame

colors = {
    'BLACK': (0, 0, 0),
    'BLUE': (100, 100, 255),
    'RED': (255, 0, 0),
    'WHITE': (255, 255, 255),
    'GREEN': (0, 255, 0),
}

def print_drone_state(drone):
    print(f"Drone Position: ({drone.x:.1f}, {drone.y:.1f}), Rotation: {drone.rotation:.1f}", end=None)
    for i in range(len(drone.motors)):
        print(f"| Motor {i} thrust: {drone.motors[i].thrust}", end=None)


def handle_user_inputs(drone, config):
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key in [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4]:
                motor_index = event.key - pygame.K_1
                if motor_index < len(drone.motors):
                    drone.motors[motor_index].apply_thrust(config["drone"]["thrust"])
        elif event.type == pygame.KEYUP:
            if event.key in [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4]:
                motor_index = event.key - pygame.K_1  # Assuming keys 1-4 for motors
                if motor_index < len(drone.motors):
                    drone.motors[motor_index].apply_thrust(0)