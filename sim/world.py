import pygame

class World:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.beacon_pos = (width // 2, height // 2)
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
        pygame.draw.circle(win, colors["beacon"], self.beacon_pos, 8)

    def collision(self, rect):
        return any(rect.colliderect(obs) for obs in self.obstacles)
