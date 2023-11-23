import pygame
import sys
import yaml

from src.drone import Motor, Drone
from src.screen import update_screen, draw_drone, draw_drone_path
from src.utilities import (
    colors,
    print_drone_state,
    handle_user_inputs,
)

def main(config: dict):
    pygame.init()
    screen = pygame.display.set_mode(
        (config["screen"]["width"], config["screen"]["height"])
    )
    pygame.display.set_caption("Drone Simulation")

    drone = Drone(
        x=config["screen"]["width"] // 2,
        y=config["screen"]["height"] // 2,
        update_frequency=config["screen"]["refresh_rate"],
    )
    for motor in config["drone"]["motors"]:
        drone.add_motor(Motor(x=motor[0], y=motor[1], rotation=motor[2], thrust=config["drone"]["thrust"]))

    running = True
    while running:
        handle_user_inputs(drone, config)

        drone.update_position()
        drone.velocity_y -= config["environment"]["gravity"]

        screen.fill(colors["WHITE"])
        draw_drone(drone, screen)
        draw_drone_path(drone, screen, config)
        update_screen()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    # Reading YAML configuration file
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    print(config)
    main(config)
