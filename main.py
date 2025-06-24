import pygame
import sys
import math
import random
import numpy as np
from filters.particle_filter import ParticleFilter
from filters.ekf import EKF
from filters.agf import AGF
from sim.agent import Agent
from sim.world import World
import matplotlib.pyplot as plt
import time

# --- Pygame Setup ---
pygame.init()
WIDTH, HEIGHT = 1200, 800
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Bayesian Filter Sim")
show_particles = True
show_ekf = True
show_grid = True

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
filter_interval = 18
# number of particles
N_s = 200
sensor_noise = 20.0
# for manual control
speed = 12
num_beacons = 1
# manual input or not
manual_control = False
# initial state of mu (if using nonmanual)
initial_vx = random.gauss(2000, 200)
initial_vy = random.gauss(2000, 200)
mu = np.array([initial_vx, initial_vy])
grid_resolution = 20

# Initialize world, agent, filter
world = World(WIDTH, HEIGHT, num_beacons)
agent = Agent(start_x=700, start_y=250, frame_rate=60, WIDTH=WIDTH, HEIGHT=HEIGHT, radius=10, speed=speed)
pf = ParticleFilter(N_s=N_s, width=WIDTH, height=HEIGHT)
ekf = EKF(np.array([agent.rect.x, agent.rect.y]), np.diag([100, 100]), world.beacons, sensor_noise)
agf = AGF(WIDTH, HEIGHT, grid_resolution, [agent.rect.x, agent.rect.y])

ekf_mean = pygame.Rect(agent.rect.x, agent.rect.y, 5, 5)

error_list_ekf = list()
error_list_agf = list()
error_list_pf = list()

clock = pygame.time.Clock()
last_update = 0
running = True

# track time steps
k = 0
T = 400

while running:
    clock.tick(frame_rate)
    win.fill(colors["background"])

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                show_particles = not show_particles
            if event.key == pygame.K_w:
                show_ekf = not show_ekf
            if event.key == pygame.K_e:
                show_grid = not show_grid

    # this is for manual control of the system
    keys = pygame.key.get_pressed()
    mu = agent.move(keys, world, mu, manual_control=manual_control)

    # Draw environment
    world.draw(win, colors)
    pygame.draw.circle(win, colors["agent"], agent.rect.center, dot_radius)

    # Sensor measurement
    true_pos = agent.get_position()
    deltas = world.beacons - true_pos
    distances = np.linalg.norm(deltas, axis = 1)
    z_k = distances + np.random.normal(0, sensor_noise, size=len(world.beacons))

    # compute recusion at the discrete time interval
    now = pygame.time.get_ticks()
    if now - last_update >= filter_interval:
        k += 1
        dt = filter_interval / 1000.0
        last_update = now

        start = time.perf_counter()
        ekf.predict(mu, dt)
        ekf.update(z_k)
        ekf_mean.center = ekf.get_state()
        ekf_time = time.perf_counter() - start

        start = time.perf_counter()
        pf.predict(mu, dt)
        pf.update(z_k, world.beacons, sensor_noise)
        pf_time = time.perf_counter() - start

        start = time.perf_counter()
        agf.predict(mu, dt)
        agf.update(z_k, world.beacons, sensor_noise)
        agf_time = time.perf_counter() - start

        print(f"EKF: {ekf_time*1000:.2f}ms | PF: {pf_time*1000:.2f}ms | Grid: {agf_time*1000:.2f}ms")

        # We resample if it drops below a threshold N_T as degeneracy is high
        N_T = N_s // 3
        if pf.effective_sample_size() < N_T:
            pf.resample()
        
        # lets metric-ify the sim
        ground_truth = true_pos
        ekf_predicted_state = ekf.get_state()
        pf_predicted_state = pf.get_estimated_state()
        agf_predicted_state = agf.get_estimated_state()
        error_ekf = np.linalg.norm(ground_truth - ekf_predicted_state)
        error_agf = np.linalg.norm(ground_truth - agf_predicted_state)
        error_pf = np.linalg.norm(ground_truth - pf_predicted_state)
        error_list_ekf.append(error_ekf)
        error_list_agf.append(error_agf)
        error_list_pf.append(error_pf)

    glow_surface.fill((0, 0, 0, 0))

    if show_particles:
        max_weight = max(p.weight for p in pf.particles) + 1e-6
        for p in pf.particles:
            normalized = p.weight / max_weight
            size = int(1 + 6 * normalized)
            pygame.draw.circle(win, colors["particle"], (int(p.x), int(p.y)), size)

            # display a weighted average of the particles
            pygame.draw.circle(win, (255, 255, 0), pf_predicted_state, dot_radius)

    if show_grid:
        grid_weights = agf.get_grid()
        grid_dimensions = agf.get_dimensions()
        max_weight = np.max(grid_weights)

        if max_weight > 0:
            for i in range(grid_dimensions[0]):
                for j in range(grid_dimensions[1]):
                    normalized = grid_weights[i, j] / max_weight
                    alpha = int(normalized * 255)
                    color = (255, 255, 255, alpha)  # white glow with variable alpha

                    x = j * grid_resolution
                    y = i * grid_resolution

                    # Draw with alpha blending
                    pygame.draw.rect(glow_surface, color, pygame.Rect(x, y, grid_resolution, grid_resolution))
        
        # display weighted average of the grid cells
        pygame.draw.circle(win, (255, 0, 255), agf_predicted_state, dot_radius)

    if show_ekf:
        # draw mean of ekf
        pygame.draw.circle(win, (0, 255, 255), ekf_mean.center, dot_radius)
        # visualize ekf, bounded by 2std from mean (95% confidence)
        sigma_x = np.sqrt(ekf.get_covariance()[0, 0])
        sigma_y = np.sqrt(ekf.get_covariance()[1, 1])
        radius_x = 2 * sigma_x
        radius_y = 2 * sigma_y

        # draw ellipse within 2 std from the mean
        pygame.draw.ellipse(
            glow_surface, 
            (255, 0, 255, 60), 
            pygame.Rect(
                ekf.get_state()[0] - radius_x, 
                ekf.get_state()[1] - radius_y, 
                2 * radius_x, 
                2 * radius_y
            ), 
            width=0 
        )
    
    pygame.draw.line(glow_surface, (0, 150, 255, 80), world.beacons[0], true_pos, width=4)
    if num_beacons == 2:
        pygame.draw.line(glow_surface, (0, 150, 255, 80), world.beacons[1], true_pos, width=4)
    if num_beacons == 3:
        pygame.draw.line(glow_surface, (0, 150, 255, 80), world.beacons[1], true_pos, width=4)
        pygame.draw.line(glow_surface, (0, 150, 255, 80), world.beacons[2], true_pos, width=4)

    win.blit(glow_surface, (0, 0))

    pygame.display.update()

    # terminate sim at threshold T. 
    if k == T:
        break

pygame.quit()

timesteps = list(range(len(error_list_ekf)))

plt.plot(timesteps, error_list_ekf, label='extended kalman filter')
plt.plot(timesteps, error_list_pf, label='bootstrap particle filter')
plt.plot(timesteps, error_list_agf, label='grid-based filter')

def compute_rmse(errors):
    return np.sqrt(np.mean(np.square(errors)))

rmse_ekf = compute_rmse(error_list_ekf)
rmse_pf = compute_rmse(error_list_pf)
rmse_agf = compute_rmse(error_list_agf)

rmse_text = (
    "RMSE: \n"
    f"EKF  : {rmse_ekf:.2f}\n"
    f"PF   : {rmse_pf:.2f}\n"
    f"Grid : {rmse_agf:.2f}"
)

plt.gca().text(
    0.98, 0.02,
    rmse_text,
    fontsize=9,
    ha='right',
    va='bottom',
    transform=plt.gca().transAxes,
    bbox=dict(facecolor='white', edgecolor='gray', alpha=0.75)
)

plt.title("a simple graphical comparison of bayesian estimators.")
plt.xlabel("discrete time step k")
plt.ylabel("distance from ground truth")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()  # This will block, but now it’s fine since the sim is done

sys.exit()