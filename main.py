import pygame
import sys
import math
import random
import numpy as np
from filters.particle_filter import ParticleFilter
from sim.agent import Agent
from sim.world import World

# --- Pygame Setup ---
pygame.init()
WIDTH, HEIGHT = 1200, 800
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Bayesian Filter Sim")

glow_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

# Colors
colors = {
    "background": (0, 200, 100),
    "obstacle": (100, 100, 100),
    "beacon": (0, 0, 255),
    "particle": (255, 0, 0),
    "estimate": (30, 30, 30),
    "agent": (0, 0, 0)
}

# Simulation Parameters
dot_radius = 10
frame_rate = 60
# run at 20 hz, 50ms time steps
filter_interval = 50
N_s = 150
sensor_noise = 25.0
speed = 8

# Initialize world, agent, filter
world = World(WIDTH, HEIGHT)
agent = Agent(start_x=700, start_y=250, frame_rate=60, WIDTH=WIDTH, HEIGHT=HEIGHT, radius=10, speed=speed)
pf = ParticleFilter(N_s=N_s, width=WIDTH, height=HEIGHT)
predicted_state = pygame.Rect(agent.rect.x, agent.rect.y, 5, 5)

clock = pygame.time.Clock()
last_update = 0
running = True

while running:
    clock.tick(frame_rate)
    win.fill(colors["background"])

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    mu = agent.move(keys, world)

    # Draw environment
    world.draw(win, colors)
    pygame.draw.circle(win, colors["agent"], agent.rect.center, dot_radius)

    # Sensor measurement
    true_pos = agent.get_position()
    z_k = math.hypot(true_pos[0] - world.beacon_pos[0], true_pos[1] - world.beacon_pos[1]) + random.gauss(0, sensor_noise)

    # Particle filter update
    now = pygame.time.get_ticks()
    if now - last_update >= filter_interval:
        dt = filter_interval / 1000.0
        last_update = now
        pf.predict(mu, dt)
        pf.update(z_k, world.beacon_pos, sensor_std=sensor_noise)
        # We resample if it drops below a threshold N_T as degeneracy is high
        N_T = N_s // 3
        if pf.effective_sample_size() < N_T:
            pf.resample()
        x_est, y_est = pf.estimate()
        predicted_state.center = (int(x_est), int(y_est))

    # draw particles 
    for p in pf.particles:
        size = max(1, min(8, int(4 * p.weight * N_s)))
        pygame.draw.circle(win, colors["particle"], (int(p.x), int(p.y)), size)

    glow_surface.fill((0, 0, 0, 0))
    # Color = light blue with some transparency (RGBA)
    glow_color = (0, 150, 255, 80)  # last value is alpha (0–255)

    pygame.draw.line(glow_surface, glow_color, world.beacon_pos, true_pos, width=4)

    win.blit(glow_surface, (0, 0))

    # pygame.draw.circle(win, colors["estimate"], predicted_state.center, 5)

    pygame.display.update()

pygame.quit()
sys.exit()
