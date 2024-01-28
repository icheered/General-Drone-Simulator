#
#   The display class is used to show a pygame screen and render the drone and target on it.
#

import pygame
import math
import random

pygame.init()

BLACK = (50,50, 50)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0,255,0)
DRONE_SIZE = 40  # Size of the drone square
MOTOR_SIZE = 40   # Size of the motor squares

# Particle class
class Particle:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.x_velocity = random.uniform(-1, 1)  # Horizontal velocity
        self.y_velocity = random.uniform(-2, 0)  # Vertical velocity (upward)
        self.lifetime = random.randint(20, 50)  # How long the particle will live
        self.size = random.randint(2, 4)  # Size of the particle

    def update(self):
        self.x += self.x_velocity
        self.y += self.y_velocity
        self.lifetime -= 1  # Decrease the lifetime
        self.size -= 0.1  # Shrink the particle
        self.size = max(self.size, 0)  # Make sure size doesn't go negative

    def draw(self, surface):
        if self.lifetime > 0:
            pygame.draw.circle(surface, (255, 255, 255), (int(self.x), int(self.y)), int(self.size))


class Display:
    def __init__(self, config: dict, title: str):
        self.width = config["display"]["width"]
        self.height = config["display"]["height"]
        self.update_frequency = config["display"]["update_frequency"]

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()

    def _draw_start_point(self, drone):
        # Scale the position to the screen size
        #print(f"Start from display: {drone.start_position}")
        startx = drone.start_position[0]
        starty = drone.start_position[1]
        ax = startx * self.width/2 + self.width/2
        ay = starty * self.height/2 + self.height/2
        pygame.draw.circle(self.screen, GREEN, (ax, ay), 5)  # Green circle for point A

    def _draw_drone(self, drone):
        # Drone state
        state = drone.get_observation(state=True, domain_params=False, targets=False)
        drone_x, _, drone_y, _, rotation, _ = state[:6]

        # drone_x and drone_y are (-1,1) so we need to scale them to the screen size
        drone_x = drone_x * self.width/2 + self.width/2
        drone_y = drone_y * self.height/2 + self.height/2

        drone_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)  # Use SRCALPHA for transparency
        
        # Fill the surface with light gray pixels
        # drone_surface.fill((10, 10, 10))

        # Draw the drone rectangle on the surface
        # drone_rect = pygame.Rect(self.width/2 - DRONE_SIZE/2, self.height/2 - DRONE_SIZE/2, DRONE_SIZE, DRONE_SIZE)
        # pygame.draw.rect(drone_surface, WHITE, drone_rect)

        action = drone.last_action

        for i, motor in enumerate(drone.motors):
            motor_x, motor_y, _, _ = motor
            # Draw a line from the motor center to the drone center
            body_height = 10
            pygame.draw.line(drone_surface, WHITE, (motor_x * 100 + self.width/2, motor_y * 100 + self.height/2), (self.width/2, self.height/2), body_height)


            # Load the image just once, best to do this outside of the draw loop
            original_target = pygame.image.load('media/rocket.png').convert_alpha()  # Ensure the image supports transparency

            # Scale the image to the new size
            rocket_size = (MOTOR_SIZE, MOTOR_SIZE)  # Desired size
            rocket = pygame.transform.scale(original_target, rocket_size)

        for i, motor in enumerate(drone.motors):
            # Calculate the position to blit the motor triangle
            motor_x, motor_y, _, _ = motor

            motor_x_scaled = motor_x * 100 + self.width/2 - MOTOR_SIZE/2
            motor_y_scaled = motor_y * 100 + self.height/2 - MOTOR_SIZE

            # Create a surface for the motor
            motor_surface = pygame.Surface((MOTOR_SIZE, MOTOR_SIZE), pygame.SRCALPHA)  # Use SRCALPHA for transparency
            motor_surface.blit(rocket, (0,0))

            # Emit particles from the rocket if activated
            action = [1, 1, 0]
            

            # # # # # Create a triangle for the motor
            # # # # color = GREEN if action[i] else RED
            # # # # motor_triangle = pygame.draw.polygon(motor_surface, color, [(0, MOTOR_SIZE), (MOTOR_SIZE/2, 0), (MOTOR_SIZE, MOTOR_SIZE)])

            # # # # # Create the number for the motor
            # # # # font = pygame.font.SysFont(None, 20)
            # # # # text_surface = font.render(str(i+1), False, (0, 0, 0)) 
            
            # # # # # Blit at the center
            # # # # text_x = MOTOR_SIZE/2 - text_surface.get_width()/2
            # # # # text_y = MOTOR_SIZE/2 - text_surface.get_height()/4

            # # # # motor_surface.blit(text_surface, (text_x, text_y))

            # Rotate the motor triangle
            motor_rotated = pygame.transform.rotate(motor_surface, (-motor[2])) # 0 degrees is right, 90 degrees is down

            # Blit the motor triangle onto the drone surface at the calculated position
            drone_surface.blit(motor_rotated, (motor_x_scaled, motor_y_scaled + MOTOR_SIZE/2))

        # Rotate the combined drone and motors surface
        rotation = rotation * 180 / math.pi
        rotated_drone_surface = pygame.transform.rotate(drone_surface, -rotation)

        # Calculate the center position for blitting
        blit_x = drone_x - rotated_drone_surface.get_width() / 2
        blit_y = drone_y - rotated_drone_surface.get_height() / 2

        # Draw the rotated drone surface on the screen
        self.screen.blit(rotated_drone_surface, (blit_x, blit_y))

    
    def _draw_state(self, drone):
        state = drone.get_observation(state=True, domain_params=False, targets=False)
        font = pygame.font.SysFont(None, 24)
        y_offset = 0  # Starting y position for the first line of text
        x_offset = 20  # Starting x position for the first line of text
        line_height = 25  # Height of each line of text

        state_labels = ["X", "vX", "Y", "vY", "angle", "vAngle"]
        domain_labels = ["mass", "inertia", "gravity"]

        # State
        text = font.render("State:", True, WHITE)
        self.screen.blit(text, (0, y_offset))
        y_offset += line_height
    
        # Draw the state
        for label, value in zip(state_labels, state):
            text = font.render(f"{label}: {round(value, 2)}", True, WHITE)
            self.screen.blit(text, (x_offset, y_offset))
            y_offset += line_height
        
        # Domain parameters
        y_offset += line_height
        text = font.render("Domain parameters:", True, WHITE)
        self.screen.blit(text, (0, y_offset))
        y_offset += line_height
        domain_parameters = drone.get_observation(state=False, domain_params=True, targets=False)

        for label, value in zip(domain_labels, domain_parameters):
            text = font.render(f"{label}: {round(value, 2)}", True, WHITE)
            self.screen.blit(text, (x_offset, y_offset))
            y_offset += line_height

        # Targets
        y_offset += line_height
        text = font.render("Targets:", True, WHITE)
        self.screen.blit(text, (0, y_offset))

        # Draw the target distances, defined as x and y times the number of targets
        # The targets come in pairs of 2, and they are the last 2*len(targets) elements of the state
        y_offset += line_height
        target_idx = len(state) - len(drone.targets)  # Starting index of targets in the state array
        for i in range(0, len(drone.targets), 2):
            x_target = state[target_idx + i]
            y_target = state[target_idx + i + 1]
            label = f"T{i//2 + 1}"
            text = font.render(f"{label}: ({round(x_target, 2)}, {round(y_target, 2)})", True, WHITE)
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

    def _draw_targets(self, drone):
        # Load the image just once, best to do this outside of the draw loop
        original_target = pygame.image.load('media/target.png').convert_alpha()  # Ensure the image supports transparency

        # Scale the image to the new size
        target_px = 30
        target_size = (target_px, target_px)  # Desired size
        target = pygame.transform.scale(original_target, target_size)

        # Tint the image red (assuming the original image is white or grayscale)
        # If the image has multiple colors, this will blend them with red, potentially leading to undesired results
        red_color = (255, 0, 0, 255)  # Red with full alpha
        target.fill(red_color, special_flags=pygame.BLEND_RGBA_MULT)

        for i in range(0, len(drone.targets), 2):
            target_x = int(drone.targets[i] * self.width / 2 + self.width / 2)
            target_y = int(drone.targets[i+1] * self.height / 2 + self.height / 2)

            # Blit the image at the calculated position
            self.screen.blit(target, (target_x - target_size[0] // 2, target_y - target_size[1] // 2))  # Center the target

    def _draw_simulation_stats(self, drone):
        # On the right hand side of the screen
        # Draw the current frame, the frames without target, the last reward
        font = pygame.font.SysFont(None, 24)
        y_offset = 20
        x_offset = 20

        text = font.render(f"Frame: {drone.episode_step}", True, WHITE)
        self.screen.blit(text, (self.width - text.get_width() - x_offset, y_offset))
        y_offset += 25

        text = font.render(f"Frames without target: {drone.episodes_without_target}", True, WHITE)
        self.screen.blit(text, (self.width - text.get_width() - x_offset, y_offset))
        y_offset += 25

        text = font.render(f"Reward: {round(drone.last_reward, 2)}", True, WHITE)
        self.screen.blit(text, (self.width - text.get_width() - x_offset, y_offset))
        y_offset += 25

    def update(self, drone):
        if not drone.enable_rendering:
            return
        self.clock.tick(60)
        self.screen.fill(BLACK)
        self._draw_drone(drone)
        self._draw_targets(drone)
        
        #self._draw_start_point(drone)
        self._draw_state(drone)
        #self._draw_simulation_stats(drone)
        pygame.display.flip()

        # Handle the event queue
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

    def close(self):
        pygame.quit()