#
#   The display class is used to show a pygame screen and render the drone and target on it.
#

import pygame
import math
import random
import colorsys
from pygame import gfxdraw

pygame.init()

BACKGROUND = "#202020"
TEXT = "#dddddd"
PRIMARY = "#33ff33"
SECONDARY = "#ff9933"
DRONE_BODY= "#3333ff"

DRONE_SIZE = 40  # Size of the drone square
MOTOR_SIZE = 40   # Size of the motor squares



# Function to convert HSL to RGB
def hsl_to_rgb(h, s, l):
    r, g, b = colorsys.hls_to_rgb(h / 360, l / 100, s / 100)
    return int(r * 255), int(g * 255), int(b * 255)

# Particle class
class Particle:
    def __init__(self, x, y, vx, vy, thrust=1):
        self.x = x + random.uniform(0,2) * vx
        self.y = y + random.uniform(0,2) * vy
        self.x_velocity = vx + random.uniform(-0.5, 0.5)  # Horizontal velocity
        self.y_velocity = vy + random.uniform(-0.5, 0.5)  # Vertical velocity (upward)
        self.thrust = thrust
        self.lifetime = random.randint(10, 20)  # How long the particle will live
        self.size = random.randint(2, 4)  # Size of the particle

    def update(self):
        self.x += self.x_velocity
        self.y += self.y_velocity
        self.lifetime -= 1  # Decrease the lifetime
        self.size -= 0.1  # Shrink the particle
        self.size = max(self.size, 0)  # Make sure size doesn't go negative

    def draw(self, surface):
        if self.lifetime > 0:
            # Gradient from light-yellow to red based on thrust
            color = hsl_to_rgb(60 - 60*self.thrust, 100, 60)
            pygame.draw.circle(surface, color, (int(self.x), int(self.y)), int(self.size))

particles = []
def emit_particles(particles, emit_position, emit_velocity, thrust, num_particles=10):
    for _ in range(num_particles):
        particles.append(Particle(*emit_position, *emit_velocity, thrust))

def draw_progress_bar(screen, position, size, progress):
    """
    Draws a progress bar on the given screen.

    :param screen: Pygame surface to draw on.
    :param position: Tuple (x, y) for the top-left position of the progress bar.
    :param size: Tuple (width, height) for the size of the progress bar.
    :param progress: Float between 0 and 1, where 0 is empty and 1 is full.
    """
    # Colors
    background_color = BACKGROUND
    fill_color = SECONDARY

    # Draw the background
    width, height = size
    pos_x, pos_y = position
    border_width = 2
    gap = 2

    # Draw green border around the progress bar
    pygame.draw.rect(screen, fill_color, (pos_x - (border_width + gap), pos_y - (border_width + gap), width + 2*(border_width + gap), height + 2*(border_width + gap)))
    # Add black background
    pygame.draw.rect(screen, background_color, (pos_x - (gap), pos_y - ( gap), width + 2*(gap), height + 2*(gap)))

    # Calculate the width of the filled area
    fill_width = size[0] * progress

    # Draw the filled area
    pygame.draw.rect(screen, fill_color, (*position, fill_width, size[1]))

def hex_to_rgb(hex):
  return tuple(int(hex[i:i+2], 16) for i in (0, 2, 4))

# Function to draw a rounded toggle
def draw_toggle(screen, position, size, is_on):
    # Colors
    base_color_on = SECONDARY  # Light green color for "on" state
    base_color_off = TEXT  # Grey color for "off" state
    circle_color = BACKGROUND  # White color for the toggle circle

    # Toggle base dimensions
    base_rect = pygame.Rect(position, size)

    # Toggle radius is half the height of the base
    gap = 2
    toggle_radius = round(size[1] / 2) - gap

    # Determine base color based on state
    base_color = base_color_on if is_on else base_color_off

    # Draw the base with rounded corners
    pygame.draw.rect(screen, base_color, base_rect, border_radius=toggle_radius)

    # Draw the circle on the base
    circle_pos_x = position[0] + size[0] - toggle_radius - gap if is_on else position[0] + toggle_radius + gap
    circle_pos = (circle_pos_x, position[1] + size[1] / 2)
    pygame.draw.circle(screen, circle_color, circle_pos, toggle_radius)


class Display:
    def __init__(self, config: dict, title: str):
        self.width = config["display"]["width"]
        self.height = config["display"]["height"]
        self.update_frequency = config["display"]["update_frequency"]

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        self.highscore = 0
    
    def reset(self):
        self.screen.fill(BACKGROUND)
        pygame.display.flip()
        particles.clear()

    def _draw_start_point(self, drone):
        # Scale the position to the screen size
        #print(f"Start from display: {drone.start_position}")
        startx = drone.start_position[0]
        starty = drone.start_position[1]
        ax = startx * self.width/2 + self.width/2
        ay = starty * self.height/2 + self.height/2
        pygame.draw.circle(self.screen, SECONDARY, (ax, ay), 5)  # Green circle for point A

    def _draw_drone(self, drone):
        # Drone state
        state = drone.get_observation(state=True, domain_params=False, targets=False)
        drone_x, _, drone_y, _, rotation, _ = state[:6]

        # drone_x and drone_y are (-1,1) so we need to scale them to the screen size
        drone_x = drone_x * self.width/2 + self.width/2
        drone_y = drone_y * self.height/2 + self.height/2

        drone_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)  # Use SRCALPHA for transparency

        for i, motor in enumerate(drone.motors):
            motor_x, motor_y, _, _ = motor
            # Draw a line from the motor center to the drone center
            body_height = 10
            pygame.draw.line(drone_surface, DRONE_BODY, (motor_x * 100 + self.width/2, motor_y * 100 + self.height/2), (self.width/2, self.height/2), body_height)

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

    def _draw_particles(self, drone):
        action = drone.last_action
        drone_x, _, drone_y, _, rotation, _ = drone.get_observation(state=True, domain_params=False, targets=False)
 
        # Drone x and y are [-1, 1], scale to pixel location
        drone_x_px = drone_x * self.width/2 + self.width/2
        drone_y_px = drone_y * self.height/2 + self.height/2

        for i, motor in enumerate(drone.motors):
            if action[i] == 0:
                continue
            
            # Calculate the position to blit the motor triangle
            motor_x, motor_y, motor_rotation, _ = motor

            # Scale the motor position to pixel location. Account for rotation.
            motor_x_px = drone_x_px + motor_x * 100 * math.cos(rotation) - motor_y * math.sin(rotation) * 100
            motor_y_px = drone_y_px + motor_x * 100 * math.sin(rotation) + motor_y * math.cos(rotation) * 100

            motor_rotation_rad = math.radians(motor_rotation)
            vx = math.cos(rotation + motor_rotation_rad + math.pi/2)  * 10
            vy = math.sin(rotation + motor_rotation_rad + math.pi/2)  * 10

            emit_particles(particles, (motor_x_px, motor_y_px), (vx, vy), action[i])

        for particle in particles[:]:
            particle.update()
            if particle.lifetime <= 0:
                particles.remove(particle)
            else:
                particle.draw(self.screen)

    def _draw_state(self, drone):
        state = drone.get_observation(state=True, domain_params=False, targets=False)
        font = pygame.font.SysFont(None, 24)
        y_offset = 350  # Starting y position for the first line of text
        x_offset = 20  # Starting x position for the first line of text
        line_height = 25  # Height of each line of text

        state_labels = ["X", "vX", "Y", "vY", "angle", "vAngle"]
        domain_labels = ["mass", "inertia", "gravity"]

        # State
        text = font.render("State:", True, TEXT)
        self.screen.blit(text, (0, y_offset))
        y_offset += line_height
    
        # Draw the state
        for label, value in zip(state_labels, state):
            text = font.render(f"{label}: {round(value, 2)}", True, TEXT)
            self.screen.blit(text, (x_offset, y_offset))
            y_offset += line_height
        
        # Domain parameters
        y_offset += line_height
        text = font.render("Domain parameters:", True, TEXT)
        self.screen.blit(text, (0, y_offset))
        y_offset += line_height
        domain_parameters = drone.get_observation(state=False, domain_params=True, targets=False)

        for label, value in zip(domain_labels, domain_parameters):
            text = font.render(f"{label}: {round(value, 2)}", True, TEXT)
            self.screen.blit(text, (x_offset, y_offset))
            y_offset += line_height

        # Targets
        y_offset += line_height
        text = font.render("Targets:", True, TEXT)
        self.screen.blit(text, (0, y_offset))

        # Draw the target distances, defined as x and y times the number of targets
        # The targets come in pairs of 2, and they are the last 2*len(targets) elements of the state
        y_offset += line_height
        targets = drone.get_observation(state=False, domain_params=False, targets=True)
        for i in range(0, len(targets), 2):
            x_target = targets[i]
            y_target = targets[i + 1]
            label = f"T{i//2 + 1}"
            text = font.render(f"{label}: ({round(x_target, 2)}, {round(y_target, 2)})", True, TEXT)
            self.screen.blit(text, (x_offset, y_offset))
            y_offset += line_height
        

    def _draw_agent_state(self, agent):
        font = pygame.font.SysFont(None, 24)
        y_offset = 20
        x_offset = 20

        text = font.render(f"Game: {agent.n_games}", True, TEXT)
        self.screen.blit(text, (self.width - text.get_width() - x_offset, y_offset))

        y_offset += 25

        text = font.render(f"Epsilon: {round(agent.epsilon*100, 1)}", True, TEXT)
        self.screen.blit(text, (self.width - text.get_width() - x_offset, y_offset))


    def _draw_action(self, action):
        # Draw at the bottom left
        font = pygame.font.SysFont(None, 24)
        y_offset = self.height - 150
        line_height = 25

        text = font.render("Action:", True, TEXT)
        self.screen.blit(text, (0, y_offset))
        y_offset += line_height

        for i, value in enumerate(action):
            text = font.render(f"{i}: {round(value * 100)}", True, TEXT)
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
        target.fill(PRIMARY, special_flags=pygame.BLEND_RGBA_MULT)

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

        text = font.render(f"Frame: {drone.episode_step}", True, TEXT)
        self.screen.blit(text, (self.width - text.get_width() - x_offset, y_offset))
        y_offset += 25

        text = font.render(f"Frames without target: {drone.episodes_without_target}", True, TEXT)
        self.screen.blit(text, (self.width - text.get_width() - x_offset, y_offset))
        y_offset += 25

        text = font.render(f"Frame reward: {round(drone.last_reward, 2)}", True, TEXT)
        self.screen.blit(text, (self.width - text.get_width() - x_offset, y_offset))
        y_offset += 25

        text = font.render(f"Total reward: {round(drone.total_reward, 1)}", True, TEXT)
        self.screen.blit(text, (self.width - text.get_width() - x_offset, y_offset))
        y_offset += 25

    
    def _draw_title(self, drone):

        if drone.hit_targets > self.highscore:
            self.highscore = drone.hit_targets
        # Print "Current Score"
        font = pygame.font.Font("media/LilitaOne-Regular.ttf", 50)
        text = font.render(f"Current Score: {drone.hit_targets}", True, TEXT)
        self.screen.blit(text, (self.width/2 - text.get_width()/2, 20))

        # Print "High score"
        font = pygame.font.Font("media/LilitaOne-Regular.ttf", 15)
        text = font.render(f"High score: {self.highscore}", True, TEXT)
        self.screen.blit(text, (self.width/2 - text.get_width()/2, 80))
        
        # Print "Simulation progress"
        font = pygame.font.Font("media/LilitaOne-Regular.ttf", 15)
        text = font.render(f"Simulation progress", True, TEXT)
        self.screen.blit(text, (self.width/2 - text.get_width()/2, 100))

        

        # Print progress bar
        progress = drone.episode_step / drone.max_episode_steps
        draw_progress_bar(self.screen, (self.width/2 - 100, 125), (200, 20), progress)
    
    def _draw_run_info(self, drone):
        font = pygame.font.Font("media/LilitaOne-Regular.ttf", 25)
        text = font.render("Settings", True, TEXT)
        self.screen.blit(text, (20, 20))


        settings = [
            drone.environment["domain_randomization"],
            drone.environment["domain_knowledge"],
            drone.environment["domain_estimation"]
        ]
        
        labels = [
            "Domain Randomization",
            "Domain Knowledge",
            "Parameter Estimation"
        ]

        font = pygame.font.Font("media/LilitaOne-Regular.ttf", 15)

        x_offset = 20
        y_offset = 50
        line_height = 40

        for i, (setting, label) in enumerate(zip(settings, labels)):
            text = font.render(f"{label}", True, TEXT)
            self.screen.blit(text, (x_offset, y_offset + line_height * i + 3))
            draw_toggle(self.screen, (x_offset + 180, y_offset + line_height * i), (50, 25), is_on=setting)

    
    def _draw_domain_parameters(self, drone):
        font = pygame.font.Font("media/LilitaOne-Regular.ttf", 25)
        text = font.render("Domain Parameters", True, TEXT)
        self.screen.blit(text, (20, 180))

        # On the left, display progress bars for each domain parameter
        font = pygame.font.Font("media/LilitaOne-Regular.ttf", 15)
        y_offset = 210
        x_offset = 20
        line_height = 30

        configs = [drone.mass_config, drone.inertia_config, drone.gravity_config]
        parameters = [drone.mass, drone.inertia, drone.gravity]
        labels = ["Mass", "Inertia", "Gravity"]

        # For each config, display a progress bar filled to the current value
        for config, parameter, label in zip(configs, parameters, labels):
            text = font.render(f"{label}", True, TEXT)
            self.screen.blit(text, (x_offset, y_offset))
            y_offset += 25

            draw_progress_bar(self.screen, (x_offset + 5, y_offset), (220, 10), max((parameter - config[0]),0.001) / max((config[2]- config[0]), 0.001))
            y_offset += line_height



    def update(self, drone):
        if not drone.enable_rendering:
            return
        
        self.clock.tick(60)
        self.screen.fill(BACKGROUND)
        
        # Draw the title
        self._draw_title(drone)
        self._draw_run_info(drone)
        self._draw_domain_parameters(drone)

        # Draw the objects  on screen
        self._draw_targets(drone)
        self._draw_drone(drone)
        self._draw_particles(drone)
        
        # For debugging purposes
        #self._draw_start_point(drone)
        #self._draw_state(drone)
        #self._draw_simulation_stats(drone)
        pygame.display.flip()

        # Handle the event queue
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

    def close(self):
        pygame.quit()