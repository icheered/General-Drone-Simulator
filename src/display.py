#
#   The display class is used to show a pygame screen and render the drone and target on it.
#

import pygame
import math

pygame.init()

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0,255,0)
DRONE_SIZE = 40  # Size of the drone square
MOTOR_SIZE = 20   # Size of the motor squares

class Display:
    def __init__(self, config: dict, title: str):
        self.width = config["display"]["width"]
        self.height = config["display"]["height"]
        self.update_frequency = config["display"]["update_frequency"]

        self.target = {
            "x": config["target"]["x"],
            "y": config["target"]["y"],
            "distance": config["target"]["distance"]
        }

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()

    def _draw_drone(self, drone):
        # Drone state
        state = drone.get_state()
        drone_x, _, drone_y, _, rotation, _ = state

        # drone_x and drone_y are (-1,1) so we need to scale them to the screen size
        drone_x = drone_x * self.width/2 + self.width/2
        drone_y = drone_y * self.height/2 + self.height/2

        drone_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)  # Use SRCALPHA for transparency
        
        # Fill the surface with light gray pixels
        # drone_surface.fill((10, 10, 10))

        # Draw the drone rectangle on the surface
        drone_rect = pygame.Rect(self.width/2 - DRONE_SIZE/2, self.height/2 - DRONE_SIZE/2, DRONE_SIZE, DRONE_SIZE)
        pygame.draw.rect(drone_surface, WHITE, drone_rect)

        for i, motor in enumerate(drone.motors):
            motor_x, motor_y, _, _ = motor
            # Draw a line from the motor center to the drone center
            pygame.draw.line(drone_surface, WHITE, (motor_x * 100 + self.width/2, motor_y * 100 + self.height/2), (self.width/2, self.height/2), 10)

        for i, motor in enumerate(drone.motors):
            # Calculate the position to blit the motor triangle
            motor_x, motor_y, _, _ = motor

            motor_x_scaled = motor_x * 100 + self.width/2 - MOTOR_SIZE/2
            motor_y_scaled = motor_y * 100 + self.height/2 - MOTOR_SIZE

            # Create a surface for the motor
            motor_surface = pygame.Surface((MOTOR_SIZE, MOTOR_SIZE), pygame.SRCALPHA)  # Use SRCALPHA for transparency

            # Create a triangle for the motor

            color = RED
            motor_triangle = pygame.draw.polygon(motor_surface, color, [(0, MOTOR_SIZE), (MOTOR_SIZE/2, 0), (MOTOR_SIZE, MOTOR_SIZE)])

            # Create the number for the motor
            font = pygame.font.SysFont(None, 20)
            text_surface = font.render(str(i+1), False, (0, 0, 0)) 
            
            # Blit at the center
            text_x = MOTOR_SIZE/2 - text_surface.get_width()/2
            text_y = MOTOR_SIZE/2 - text_surface.get_height()/4

            motor_surface.blit(text_surface, (text_x, text_y))

            # Rotate the motor triangle
            motor_triangle = pygame.transform.rotate(motor_surface, (-motor[2])) # 0 degrees is right, 90 degrees is down

            # Blit the motor triangle onto the drone surface at the calculated position
            drone_surface.blit(motor_triangle, (motor_x_scaled, motor_y_scaled + MOTOR_SIZE/2))

        # Rotate the combined drone and motors surface
        rotation = rotation * 180 / math.pi
        rotated_drone_surface = pygame.transform.rotate(drone_surface, -rotation)

        # Calculate the center position for blitting
        blit_x = drone_x - rotated_drone_surface.get_width() / 2
        blit_y = drone_y - rotated_drone_surface.get_height() / 2

        # Draw the rotated drone surface on the screen
        self.screen.blit(rotated_drone_surface, (blit_x, blit_y))

    
    def _draw_state(self, state, action):
        font = pygame.font.SysFont(None, 24)
        y_offset = 20  # Starting y position for the first line of text
        x_offset = 20  # Starting x position for the first line of text
        line_height = 25  # Height of each line of text

        state_labels = ["X", "vX", "Y", "vY", "angle", "vAngle"]
        # action_labels = ["motor 1", "motor 2", "motor 3"]
        action_labels = ["motor {}".format(i+1) for i in range(len(action))]

        for label, value in zip(state_labels, state):
            text = font.render(f"{label}: {round(value, 2)}", True, WHITE)
            self.screen.blit(text, (x_offset, y_offset))
            y_offset += line_height

        for label, value in zip(action_labels, action):
            text = font.render(f"{label}: {round(value, 2)}", True, WHITE)
            self.screen.blit(text, (x_offset, y_offset))
            y_offset += line_height

    def _draw_agent_state(self, agent):
        font = pygame.font.SysFont(None, 24)
        y_offset = 20
        x_offset = 20

        text = font.render(f"Game: {agent.n_games}", True, WHITE)
        self.screen.blit(text, (self.width - text.get_width() - x_offset, y_offset))

        y_offset += 25

        text = font.render(f"Epsilon: {round(agent.epsilon*100, 1)}", True, WHITE)
        self.screen.blit(text, (self.width - text.get_width() - x_offset, y_offset))


    def _draw_action(self, action):
        # Draw at the bottom left
        font = pygame.font.SysFont(None, 24)
        y_offset = self.height - 150
        line_height = 25

        text = font.render("Action:", True, WHITE)
        self.screen.blit(text, (0, y_offset))
        y_offset += line_height

        for i, value in enumerate(action):
            text = font.render(f"{i}: {round(value * 100)}", True, WHITE)
            self.screen.blit(text, (0, y_offset))
            y_offset += line_height

        
    def _draw_target(self):
        # Draw a dot and a circle around the target with radius 50
        # Scale the target position to the screen size
        target = self.target
        x = target["x"] * self.width/2 + self.width/2
        y = target["y"] * self.height/2 + self.height/2
        distance = target["distance"] * self.width/2
        
        pygame.draw.circle(self.screen, GREEN, (x, y), distance, 1)
        pygame.draw.circle(self.screen, GREEN, (x, y), 2)

    def update(self, drone):
        self.clock.tick(60)
        self.screen.fill(BLACK)
        self._draw_drone(drone)
        self._draw_target()
        self._draw_state(drone.get_state(), drone.get_action())
        # self._draw_agent_state(agent)
        pygame.display.flip()

    def close(self):
        pygame.quit()