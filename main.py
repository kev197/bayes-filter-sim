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
import os

def run_simulation(render_sim, T, seed = None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if not render_sim:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    # --- Pygame Setup ---
    pygame.init()
    WIDTH, HEIGHT = 1200, 800
    win = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Bayesian Filter Sim")
    show_particles = True
    show_ekf = True
    show_grid = True

    glow_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    grid_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

    def render_grid(grid_weights):
        max_weight = np.max(grid_weights)
        if max_weight > 0:
            for i in range(grid_weights.shape[0]):
                for j in range(grid_weights.shape[1]):
                    w = grid_weights[i, j]
                    if w == 0:
                        continue
                    normalized = w / max_weight
                    alpha = int(normalized * 255)
                    color = (255, 255, 255, alpha)
                    x = j * grid_resolution
                    y = i * grid_resolution
                    pygame.draw.rect(grid_surface, color, pygame.Rect(x, y, grid_resolution, grid_resolution))

    def draw_glow(surface, x, y, color, max_radius=20, steps=6):
        r, g, b = color
        for i in range(steps):
            alpha = int(255 * (1 - i / steps) * 0.3)  # decreasing alpha
            radius = int(max_radius * (i + 1) / steps)
            glow_color = (r, g, b, alpha)
            pygame.draw.circle(surface, glow_color, (x, y), radius)

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
    # time step lower bound, ms
    filter_interval = 20
    real_time_filtering = True
    # number of particles
    N_s = 120
    sensor_noise = 30.0
    # for manual control
    speed = 8
    num_beacons = 1
    # manual input or not
    manual_control = False
    # initial state of mu (if using nonmanual)
    initial_velocity = random.gauss(1000, 500)
    angle = random.uniform(0, 2 * math.pi)
    mu = np.array([initial_velocity * math.cos(angle), initial_velocity * math.sin(angle)])
    grid_resolution = 30

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
    error_list_true = list()

    clock = pygame.time.Clock()
    last_update = 0
    running = True

    # track time steps
    k = 0
    pf_predicted_state = agent.get_position()
    agf_predicted_state = agent.get_position()
    ekf_predicted_state = agent.get_position()
    no_error_state = agent.get_position()
    true_pos = agent.get_position()

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

        now = pygame.time.get_ticks()
        # compute recursion at the discrete time interval
        if real_time_filtering or now - last_update >= filter_interval:
            dt = (now - last_update) / 1000.0
            last_update = now
            if dt > 0.5:
                print(f"Skipping update: large dt = {dt:.2f}s (startup lag)")
                continue
            # this is for manual control of the system
            keys = pygame.key.get_pressed()
            mu, no_error_state = agent.move(keys, world, mu, dt, manual_control=manual_control)

            # Sensor measurement
            true_pos = agent.get_position()
            deltas = world.beacons - true_pos
            distances = np.linalg.norm(deltas, axis = 1)
            z_k = distances + np.random.normal(0, sensor_noise, size=len(world.beacons))
            k += 1

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

            # print(f"EKF: {ekf_time*1000:.2f}ms | PF: {pf_time*1000:.2f}ms | Grid: {agf_time*1000:.2f}ms")

            # We resample if it drops below a threshold N_T as degeneracy is high
            N_T = N_s // 1
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
            error_true = np.linalg.norm(ground_truth - no_error_state)
            error_list_ekf.append(error_ekf)
            error_list_agf.append(error_agf)
            error_list_pf.append(error_pf)
            error_list_true.append(error_true)

        # Draw environment
        world.draw(win, colors)
        pygame.draw.circle(win, colors["agent"], agent.rect.center, dot_radius)

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
            render_grid(agf.get_grid())
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
        
        pygame.draw.circle(win, (255, 215, 0), no_error_state, 10)
        # draw_glow(glow_surface, no_error_state.centerx, no_error_state.centery, (255, 215, 0))

        glow_surface.blit(grid_surface, (0, 0))
        win.blit(glow_surface, (0, 0))

        pygame.display.update()

        # terminate sim at threshold T. 
        if k == T:
            break

    pygame.quit()

    def compute_rmse(errors):
        return np.sqrt(np.mean(np.square(errors)))
    
    rmse_ekf = compute_rmse(error_list_ekf)
    rmse_pf = compute_rmse(error_list_pf)
    rmse_agf = compute_rmse(error_list_agf)
    rmse_true = compute_rmse(error_list_true)

    def plot_results():
        timesteps = list(range(len(error_list_ekf)))

        plt.plot(timesteps, error_list_ekf, label='extended kalman filter')
        plt.plot(timesteps, error_list_pf, label='bootstrap particle filter')
        plt.plot(timesteps, error_list_agf, label='grid-based filter')
        plt.plot(timesteps, error_list_true, label='unaltered state (no filter)')

        rmse_text = (
            "RMSE: \n"
            f"EKF  : {rmse_ekf:.2f}\n"
            f"PF   : {rmse_pf:.2f}\n"
            f"Grid : {rmse_agf:.2f}\n"
            f"Unaltered : {rmse_true:.2f}"
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

        plt.show()

    if render_sim:
        plot_results()

    return (
        rmse_ekf,
        rmse_pf,
        rmse_agf,
        rmse_true
    )

run_simulation(True, 300, None)