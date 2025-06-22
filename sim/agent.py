import pygame
import numpy as np
import random
import math
from sim.world import World

class Agent:
    def __init__(self, start_x, start_y, frame_rate, WIDTH, HEIGHT, radius=10, speed=5):
        self.rect = pygame.Rect(start_x, start_y, radius * 2, radius * 2)
        self.radius = radius
        self.speed = speed
        self.true_state = np.array([start_x, start_y], dtype=float)
        self.frame_rate = frame_rate
        # for ornstein-uhlenbeck process make global variables
        self.ou_x = 0.0
        self.ou_y = 0.0
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT

    def move(self, keys, world):
        # control input [vx, vy]
        mu = np.array([0, 0])

        # general error in the system dynamics
        motion_uncertainty = random.betavariate(5, 2)

        # ornstein-uhlenbeck process for "dead reckoning"
        ou_noise = random.gauss(0, 1)
        theta = 0.0001
        sigma = 10.0
        self.ou_x = -theta * self.ou_x * (1/(self.frame_rate)) + sigma * math.sqrt(1/(self.frame_rate)) * ou_noise
        self.ou_y = -theta * self.ou_y * (1/(self.frame_rate)) + sigma * math.sqrt(1/(self.frame_rate)) * ou_noise
        
        # extra noise (bumpiness or something)
        minor_uncertainty = random.gauss(-0.2, 0.4)

        # --- Try moving X axis ---
        x_rect = self.rect.copy()
        if keys[pygame.K_LEFT]:
            x_rect.x -= self.speed * motion_uncertainty + min(0, self.ou_x) + minor_uncertainty
            mu[0] = -1 * self.speed * self.frame_rate
        if keys[pygame.K_RIGHT]:
            x_rect.x += self.speed * motion_uncertainty + min(0, self.ou_x) + minor_uncertainty
            mu[0] = self.speed * self.frame_rate


        within_bounds_x = 0 <= x_rect.left and x_rect.right <= self.WIDTH
        collision_x = world.collision(x_rect)
        if within_bounds_x and not collision_x:
            self.rect.x = x_rect.x
        else:
            mu[0] *= 0.1

        # --- Try moving Y axis ---
        y_rect = self.rect.copy()
        if keys[pygame.K_UP]:
            y_rect.y -= self.speed * motion_uncertainty + min(0, self.ou_y) + minor_uncertainty
            mu[1] = -1 * self.speed * self.frame_rate
        if keys[pygame.K_DOWN]:
            y_rect.y += self.speed * motion_uncertainty + min(0, self.ou_y) + minor_uncertainty
            mu[1] = self.speed * self.frame_rate

        within_bounds_y = 0 <= y_rect.top and y_rect.bottom <= self.HEIGHT
        collision_y = world.collision(y_rect)
        if within_bounds_y and not collision_y:
            self.rect.y = y_rect.y
        else:
            mu[1] *= 0.1

        return mu

    def get_position(self):
        return self.rect.center
