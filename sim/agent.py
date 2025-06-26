import pygame
import numpy as np
import random
import math
from sim.world import World

class Agent:
    def __init__(self, start_x, start_y, frame_rate, WIDTH, HEIGHT, radius=10, speed=5):
        self.rect = pygame.Rect(start_x, start_y, radius * 2, radius * 2)
        self.no_error_rect = pygame.Rect(start_x, start_y, radius * 2, radius * 2)
        self.radius = radius
        self.speed = speed
        self.true_state = np.array([start_x, start_y], dtype=float)
        self.frame_rate = frame_rate
        # for ornstein-uhlenbeck process make global variables
        self.ou_x = -4.0
        self.ou_y = -4.0
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT

    def move(self, keys, world, mu, dt, manual_control):
        if manual_control is True:
            no_error_state = self.no_error_rect

            # control input [vx, vy]
            mu = np.array([0, 0])

            # general error in the system dynamics
            motion_uncertainty = random.betavariate(15, 2)

            # ornstein-uhlenbeck process for "dead reckoning"
            ou_noise = random.gauss(0, 1)
            theta = 0.0001
            sigma = 10.0
            self.ou_x = -theta * self.ou_x * dt + sigma * math.sqrt(dt) * ou_noise
            self.ou_y = -theta * self.ou_y * dt + sigma * math.sqrt(dt) * ou_noise
            
            # extra noise (bumpiness or something)
            minor_uncertainty_x = random.gauss(0, 1)
            minor_uncertainty_y = random.gauss(0, 1)

            # --- Try moving X axis ---
            x_rect = self.rect.copy()
            x_rect_no_error = no_error_state.copy()
            if keys[pygame.K_LEFT]:
                x_rect.x -= self.speed * motion_uncertainty + min(0, self.ou_x) + minor_uncertainty_x
                x_rect_no_error.x -= self.speed
                mu[0] = -1 * self.speed * self.frame_rate
            if keys[pygame.K_RIGHT]:
                x_rect.x += self.speed * motion_uncertainty + min(0, self.ou_x) + minor_uncertainty_y
                x_rect_no_error.x += self.speed
                mu[0] = self.speed * self.frame_rate


            within_bounds_x = 0 <= x_rect.left and x_rect.right <= self.WIDTH
            collision_x = world.collision(x_rect)
            if within_bounds_x and not collision_x:
                self.rect.x = x_rect.x
                no_error_state.x = x_rect_no_error.x
            else:
                mu[0] *= 0.1

            # --- Try moving Y axis ---
            y_rect = self.rect.copy()
            y_rect_no_error = no_error_state.copy()
            if keys[pygame.K_UP]:
                y_rect.y -= self.speed * motion_uncertainty + self.ou_y + minor_uncertainty_y
                y_rect_no_error.y -= self.speed
                mu[1] = -1 * self.speed * self.frame_rate
            if keys[pygame.K_DOWN]:
                y_rect.y += self.speed * motion_uncertainty + min(0, self.ou_y) + minor_uncertainty_y
                y_rect_no_error.y += self.speed
                mu[1] = self.speed * self.frame_rate

            within_bounds_y = 0 <= y_rect.top and y_rect.bottom <= self.HEIGHT
            collision_y = world.collision(y_rect)
            if within_bounds_y and not collision_y:
                self.rect.y = y_rect.y
                no_error_state.y = y_rect_no_error.y
            else:
                mu[1] *= 0.1

            return mu, no_error_state
        else:
            no_error_state = self.no_error_rect

            # general error in the system dynamics
            motion_uncertainty = random.betavariate(3, 2)

            # ornstein-uhlenbeck process for "dead reckoning"
            ou_noise = random.gauss(0, 1)
            theta = 0.00001
            sigma = 500.0
            self.ou_x = -theta * self.ou_x * dt + sigma * math.sqrt(dt) * ou_noise
            self.ou_y = -theta * self.ou_y * dt + sigma * math.sqrt(dt) * ou_noise
            
            # extra noise (bumpiness or something)
            minor_uncertainty_x = random.gauss(0, 10)
            minor_uncertainty_y = random.gauss(0, 10)

            # --- Try moving X axis ---
            x_rect = self.rect.copy()
            x_rect.x += (mu[0] * motion_uncertainty + min(0, self.ou_x) + minor_uncertainty_x) * dt
            no_error_state.x += mu[0] * dt

            within_bounds_x = 0 <= x_rect.left and x_rect.right <= self.WIDTH
            collision_x = world.collision(x_rect)
            if within_bounds_x and not collision_x:
                self.rect.x = x_rect.x
            else:
                stochastic_x = random.gauss(1, 0.05)
                no_error_state.x -= mu[0] * dt
                mu[0] = mu[0] * -1.0 * stochastic_x
                self.rect.x += (mu[0] * motion_uncertainty + min(0, self.ou_x) + minor_uncertainty_x) * dt
                no_error_state.x += mu[0] * dt
                

            # --- Try moving Y axis ---
            y_rect = self.rect.copy()
            y_rect.y += (mu[1] * motion_uncertainty + min(0, self.ou_y) + minor_uncertainty_y) * dt
            no_error_state.y += mu[1] * dt

            within_bounds_y = 0 <= y_rect.top and y_rect.bottom <= self.HEIGHT
            collision_y = world.collision(y_rect)
            if within_bounds_y and not collision_y:
                self.rect.y = y_rect.y
            else:
                stochastic_y = random.gauss(1, 0.05)
                no_error_state.y -= mu[1] * dt
                mu[1] = mu[1] * -1.0 * stochastic_y
                self.rect.y += (mu[1] * motion_uncertainty + min(0, self.ou_y) + minor_uncertainty_y) * dt
                no_error_state.y += mu[1] * dt

            return mu, no_error_state.center

    def get_position(self):
        return np.array([self.rect.x, self.rect.y])
    
    def get_rect(self):
        return self.rect
