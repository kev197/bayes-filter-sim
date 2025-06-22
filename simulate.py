import pygame
import sys
import math
import random
import numpy as np
import copy

class Particle:
    def __init__(self, x, y, weight=1.0):
        self.x = x
        self.y = y
        self.weight = weight

    def state(self):
        return (self.x, self.y)

    def move(self, dx, dy, noise_std):
        import random
        self.x += dx + random.gauss(0, noise_std)
        self.y += dy + random.gauss(0, noise_std)

# --- Initialize ---
pygame.init()
WIDTH, HEIGHT = 1200, 800
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("PF Sim")
glow_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

# --- Colors ---
BLACK = (0, 0, 0)
GRAY = (100, 100, 100)
GREEN = (0, 200, 100)

start_pos_x = 700
start_pos_y = 250
# --- Black Dot ---
dot_radius = 10
dot_rect = pygame.Rect(start_pos_x, start_pos_y, dot_radius * 2, dot_radius * 2)
speed = 5

# average particle position
particle_prediction = pygame.Rect(start_pos_x, start_pos_y, dot_radius * 2, dot_radius * 2)

predicted_state = pygame.Rect(start_pos_x, start_pos_y, 5, 5)

true_state = np.array([start_pos_x, start_pos_y])

beacon_outline = pygame.Rect(WIDTH // 2 - dot_radius, HEIGHT // 2 - dot_radius, dot_radius * 2, dot_radius * 2)

# --- Obstacles (rectangles) ---
obstacles = [
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

last_update = 0
# run at 20 hz, 0.5ms time steps
filter_interval = 50

# number of particles
N_s = 150

# init particles
particles = [Particle(random.uniform(0, WIDTH), random.uniform(0, HEIGHT), weight=1.0/N_s) for _ in range(N_s)]

# for ornstein-uhlenbeck process make global variables
ou_x = 0.0
ou_y = 0.0

frame_rate = 60
# --- Main Loop ---
clock = pygame.time.Clock()
running = True
while running:
    clock.tick(frame_rate)
    win.fill(GREEN)

    # control input [vx, vy]
    mu = np.array([0, 0])

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()

    # general error in the system dynamics
    motion_uncertainty = random.betavariate(5, 2)

    # ornstein-uhlenbeck process for "dead reckoning"
    ou_noise = random.gauss(0, 1)
    theta = 0.0001
    sigma = 10.0
    ou_x = -theta * ou_x * (1/(frame_rate)) + sigma * math.sqrt(1/(frame_rate)) * ou_noise
    ou_y = -theta * ou_y * (1/(frame_rate)) + sigma * math.sqrt(1/(frame_rate)) * ou_noise
    
    # extra noise (bumpiness or something)
    minor_uncertainty = random.gauss(-0.2, 0.4)

    # --- Try moving X axis ---
    x_rect = dot_rect.copy()
    x_rect_pure = predicted_state.copy()
    if keys[pygame.K_LEFT]:
        x_rect.x -= speed * motion_uncertainty + min(0, ou_x) + minor_uncertainty
        x_rect_pure.x -= speed * motion_uncertainty
        mu[0] = -1 * speed * frame_rate
        true_state[0] -= speed * motion_uncertainty + min(0, ou_x) + minor_uncertainty
    if keys[pygame.K_RIGHT]:
        x_rect.x += speed * motion_uncertainty + min(0, ou_x) + minor_uncertainty
        x_rect_pure.x += speed * motion_uncertainty
        mu[0] = speed * frame_rate
        true_state[0] += speed * motion_uncertainty + min(0, ou_x) + minor_uncertainty


    within_bounds_x = 0 <= x_rect.left and x_rect.right <= WIDTH
    collision_x = any(x_rect.colliderect(obs) for obs in obstacles)
    if within_bounds_x and not collision_x:
        dot_rect.x = x_rect.x
        predicted_state.x = x_rect_pure.x
    else:
        mu[0] *= 0.1

    # --- Try moving Y axis ---
    y_rect = dot_rect.copy()
    y_rect_pure = predicted_state.copy()
    if keys[pygame.K_UP]:
        y_rect.y -= speed * motion_uncertainty + min(0, ou_y) + minor_uncertainty
        y_rect_pure.y -= speed * motion_uncertainty
        mu[1] = -1 * speed * frame_rate
        true_state[1] -= speed * motion_uncertainty + min(0, ou_y) + minor_uncertainty
    if keys[pygame.K_DOWN]:
        y_rect.y += speed * motion_uncertainty + min(0, ou_y) + minor_uncertainty
        y_rect_pure.y += speed * motion_uncertainty
        mu[1] = speed * frame_rate
        true_state[1] += speed * motion_uncertainty + min(0, ou_y) + minor_uncertainty

    within_bounds_y = 0 <= y_rect.top and y_rect.bottom <= HEIGHT
    collision_y = any(y_rect.colliderect(obs) for obs in obstacles)
    if within_bounds_y and not collision_y:
        dot_rect.y = y_rect.y
        predicted_state.y = y_rect_pure.y
    else:
        mu[1] *= 0.1
    

    # Draw Obstacles
    for obs in obstacles:
        pygame.draw.rect(win, GRAY, obs)

    # Draw Dot
    pygame.draw.circle(win, BLACK, (int(dot_rect.center[0]), int(dot_rect.center[1])), dot_radius)

    # pygame.draw.circle(win, (0, 0, 255), (int(predicted_state.center[0]), int(predicted_state.center[1])), dot_radius)
    pygame.draw.circle(win, BLACK, beacon_outline.center, dot_radius)

    beacon_pos = (WIDTH // 2, HEIGHT // 2)
    pygame.draw.circle(win, (0, 0, 255), beacon_pos, 8)

    sensor_noise = 25.0
    true_pos = dot_rect.center
    true_dist = math.hypot(true_pos[0] - beacon_pos[0], true_pos[1] - beacon_pos[1])
    z_k = true_dist + random.gauss(0, sensor_noise)

    now = pygame.time.get_ticks()
    
    if now - last_update >= filter_interval:
        last_update = now

        # SIS (Sequential Importance Random Sampling)

        # I need to come up with an importance density. 
        # I will use the prior for this task (bootstrap filter). 
        sum_weights = 0.0
        for p in particles:
            motion_uncertainty_predict = random.betavariate(6, 3)
            # minor_uncertainty_predict_x = random.gauss(0.0, 8.0)
            # minor_uncertainty_predict_y = random.gauss(0.0, 8.0)

            angle = random.uniform(0, 2 * math.pi)
            r = random.gauss(0.0, 8.0)
            # x = x_0 + v_x*dt + noise
            # y = y_0 + v_y*dt + noise
            p.x = p.x + ((mu[0] * motion_uncertainty_predict) * (filter_interval / 1000.0)) + r * math.cos(angle)
            p.y = p.y + ((mu[1] * motion_uncertainty_predict) * (filter_interval / 1000.0)) + r * math.sin(angle)

            # update weight for each particle recursively
            z_hat = math.hypot(p.x - beacon_pos[0], p.y - beacon_pos[1])
            error = z_hat - z_k

            # should ideally match true sensor noise
            likelihood_std = 25.0
            coeff = 1.0 / (math.sqrt(2 * math.pi) * likelihood_std)
            exponent = - (error ** 2) / (2 * likelihood_std ** 2)
            likelihood = coeff * math.exp(exponent)
            p.weight = p.weight * likelihood
            sum_weights += p.weight
        # normalize particles to form valid pdf
        sum_sq_weights = 0
        for p in particles:
            p.weight = p.weight / sum_weights
            sum_sq_weights += p.weight ** 2
        
        # approximate effective sample size. 
        # if it drops below a threshold N_t as degeneracy is high
        N_eff_hat = 1 / sum_sq_weights
        N_T = N_s 
        if N_eff_hat < N_T:
            # systematic resampling as detailed by Arulampalam et al. (2002)
            cdf = [particles[0].weight]
            for i in range(1, N_s):
                cdf.append(cdf[-1] + particles[i].weight)

            # 2. Draw a starting point u1 ∈ [0, 1/N)
            u1 = random.uniform(0, 1.0 / N_s)

            # 3. Generate N equally spaced target points
            i = 0
            new_particles = []
            for j in range(N_s):
                uj = u1 + j / N_s
                while uj > cdf[i]:
                    i += 1
                # Copy the selected particle
                selected = copy.deepcopy(particles[i])
                selected.weight = 1.0 / N_s  # Reset weight after resampling
                new_particles.append(selected)
            particles = new_particles        

    sum_particles_x = 0
    sum_particles_y = 0

    # draw particles 
    for p in particles:
        size = max(1, min(8, int(4 * p.weight * N_s)))
        pygame.draw.circle(win, (255, 0, 0), (int(p.x), int(p.y)), size)

        sum_particles_x += p.x
        sum_particles_y += p.y
    
    predicted_state.x = sum_particles_x / N_s
    predicted_state.y = sum_particles_y / N_s

    glow_surface.fill((0, 0, 0, 0))
    # Color = light blue with some transparency (RGBA)
    glow_color = (0, 150, 255, 80)  # last value is alpha (0–255)

    pygame.draw.line(glow_surface, glow_color, beacon_pos, true_pos, width=4)

    win.blit(glow_surface, (0, 0))

    # draw averaged PF prediction
    # pygame.draw.circle(win, (120, 30, 30), (int(predicted_state.center[0]), int(predicted_state.center[1])), 5.0)

    pygame.display.update()

pygame.quit()
sys.exit()
