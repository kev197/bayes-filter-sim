import pygame
import numpy as np

_WALL_BODY      = ( 52,  62,  72)
_WALL_HIGHLIGHT = ( 78,  92, 108)
_WALL_SHADOW    = ( 28,  34,  40)
_DOT_GRID       = ( 32,  40,  50)
_BEACON_OUTER   = (  0, 100, 180)
_BEACON_MID     = (  0, 150, 220)
_BEACON_CORE    = (  0, 190, 255)
_BEACON_INNER   = (180, 230, 255)

class World:
    def __init__(self, width, height, num_beacons):
        self.width = width
        self.height = height
        self.num_beacons = num_beacons

        T = 18  # uniform wall thickness throughout

        self.obstacles = [
            # ── Left room ──────────────────────────────────────────────
            pygame.Rect( 80, 100, 240,   T),  # top wall
            pygame.Rect( 80, 100,   T, 320),  # left wall
            pygame.Rect( 80, 400, 160,   T),  # bottom wall  (door gap: right)
            pygame.Rect(300, 100,   T, 190),  # right wall top (door gap: below)
            pygame.Rect(300, 330,   T,  90),  # right wall bottom

            # ── Top alcove (above center) ───────────────────────────────
            pygame.Rect(420,  60,   T, 160),  # left wall
            pygame.Rect(420,  60, 200,   T),  # top wall
            pygame.Rect(620,  60,   T, 160),  # right wall

            # ── Center pillar ───────────────────────────────────────────
            pygame.Rect(530, 240, 100, 100),

            # ── Right corridor ──────────────────────────────────────────
            pygame.Rect(780, 100, 220,   T),  # top wall
            pygame.Rect(780, 100,   T, 200),  # left wall
            pygame.Rect(980, 100,   T, 300),  # right wall
            pygame.Rect(840, 380, 160,   T),  # bottom wall  (door gap: left)

            # ── Lower-left barrier ──────────────────────────────────────
            pygame.Rect(120, 560, 200,   T),
            pygame.Rect(120, 460,   T, 100),

            # ── Lower-right enclosure ───────────────────────────────────
            pygame.Rect(940, 420, 140,   T),  # top
            pygame.Rect(1060, 420,  T, 180),  # right
            pygame.Rect(880, 580, 200,   T),  # bottom
        ]

        if num_beacons == 1:
            self.beacons = np.array([[width // 2, height // 2]])
        elif num_beacons == 2:
            self.beacons = np.array([[width // 2, height // 2], [430, 220]])
        else:
            self.beacons = np.array([[width // 2, height // 2], [430, 220], [1100, 250]])

    def draw(self, win, colors):
        # Subtle dot grid over the filled background
        for x in range(0, self.width, 30):
            for y in range(0, self.height, 30):
                pygame.draw.circle(win, _DOT_GRID, (x, y), 1)

        # Walls: offset shadow → body → top-left highlight edge
        for obs in self.obstacles:
            pygame.draw.rect(win, _WALL_SHADOW, obs.move(2, 2))
            pygame.draw.rect(win, _WALL_BODY, obs)
            pygame.draw.line(win, _WALL_HIGHLIGHT, obs.topleft, obs.topright,   2)
            pygame.draw.line(win, _WALL_HIGHLIGHT, obs.topleft, obs.bottomleft, 2)

        # Beacons: concentric rings + crosshair ticks
        for i in range(self.num_beacons):
            bx, by = int(self.beacons[i][0]), int(self.beacons[i][1])
            pygame.draw.circle(win, _BEACON_OUTER, (bx, by), 22, 1)
            pygame.draw.circle(win, _BEACON_MID,   (bx, by), 15, 1)
            pygame.draw.circle(win, _BEACON_CORE,  (bx, by),  7)
            pygame.draw.circle(win, _BEACON_INNER, (bx, by),  3)
            # four tick marks just outside the inner ring
            pygame.draw.line(win, _BEACON_CORE, (bx - 22, by), (bx - 10, by), 1)
            pygame.draw.line(win, _BEACON_CORE, (bx + 10, by), (bx + 22, by), 1)
            pygame.draw.line(win, _BEACON_CORE, (bx, by - 22), (bx, by - 10), 1)
            pygame.draw.line(win, _BEACON_CORE, (bx, by + 10), (bx, by + 22), 1)

    def collision(self, rect):
        return any(rect.colliderect(obs) for obs in self.obstacles)
