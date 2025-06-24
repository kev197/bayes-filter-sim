import pygame
import numpy as np

class World:
    def __init__(self, width, height, num_beacons):
        self.width = width
        self.height = height
        self.num_beacons = num_beacons
        self.beacons = None
        if num_beacons == 1:
            self.beacons = np.array([[width // 2, height // 2]])
        elif num_beacons == 2:
            self.beacons = np.array([
                [width // 2, height // 2], 
                [430, 220]
            ])
        else:
            self.beacons = np.array([
                [width // 2, height // 2], 
                [430, 220],
                [1100,250]
            ])
        self.obstacles = [
            pygame.Rect(200, 150, 100, 300),
            pygame.Rect(500, 100, 50, 400),
            pygame.Rect(300, 400, 200, 30),
            pygame.Rect(200, 600, 180, 80),
            pygame.Rect(900, 300, 100, 200),
            pygame.Rect(700, 600, 150, 80),
            pygame.Rect(400, 200, 50, 50),
            pygame.Rect(1000, 100, 160, 300),
            pygame.Rect(570, 370, 80, 80)
        ]

    def draw(self, win, colors):
        for obs in self.obstacles:
            pygame.draw.rect(win, colors["obstacle"], obs)
        pygame.draw.circle(win, colors["beacon"], self.beacons[0], 8)
        if self.num_beacons == 2:
            pygame.draw.circle(win, colors["beacon"], self.beacons[1], 8)
        if self.num_beacons == 3:
            pygame.draw.circle(win, colors["beacon"], self.beacons[1], 8)
            pygame.draw.circle(win, colors["beacon"], self.beacons[2], 8)

    def collision(self, rect):
        return any(rect.colliderect(obs) for obs in self.obstacles)
