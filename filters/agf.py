import numpy as np
from scipy.stats import beta
from numba import njit

# use numba speed up because of high computational load of grid based filter. 

@njit
def compute_prior(curr_x, curr_y, flow_x, flow_y, mu_x, mu_y, dt, sigma=12, alpha=1.3):
    expected_x = flow_x + alpha * mu_x * dt
    expected_y = flow_y + alpha * mu_y * dt
    dx = curr_x - expected_x
    dy = curr_y - expected_y
    dist_squared = dx**2 + dy**2
    coeff = 1.0 / (2 * np.pi * sigma**2)
    exponent = -dist_squared / (2 * sigma**2)
    return coeff * np.exp(exponent)

@njit
def estimate_jit(weights, centers):
    grid_height, grid_width = weights.shape
    weighted_avg_X = 0
    weighted_avg_Y = 0
    for i in range(grid_height):
        for j in range(grid_width):
            curr_x, curr_y = centers[i, j]
            weighted_avg_X += weights[i][j] * curr_x
            weighted_avg_Y += weights[i][j] * curr_y
    return [weighted_avg_X, weighted_avg_Y]

@njit
def predict_jit(weights, centers, mu, dt, res):
    grid_height, grid_width = weights.shape
    new_weights = np.zeros_like(weights)
    mu_x, mu_y = mu
    motion_magnitude = (mu_x**2 + mu_y**2)**0.5
    motion_radius = int(motion_magnitude * dt / res) + 3

    sum = 0

    for i in range(grid_height):
        for j in range(grid_width):
            curr_x, curr_y = centers[i, j]
            total_flow = 0.0
            for di in range(-motion_radius, motion_radius + 1):
                for dj in range(-motion_radius, motion_radius + 1):
                    k = i + di
                    l = j + dj
                    if 0 <= k < grid_height and 0 <= l < grid_width:
                        flow_x, flow_y = centers[k, l]
                        weight = weights[k, l]
                        prior = compute_prior(curr_x, curr_y, flow_x, flow_y, mu_x, mu_y, dt)
                        total_flow += weight * prior
            new_weights[i, j] = total_flow
            sum += new_weights[i, j]
    return new_weights / sum

class AGF:
    def __init__(self, WIDTH, HEIGHT, resolution, start_pos):
        self.res = resolution
        self.grid_height = HEIGHT // resolution
        self.grid_width = WIDTH // resolution
        self.N_s = self.grid_width * self.grid_height
        self.centers = np.array([[(j * self.res + self.res / 2, i * self.res + self.res / 2)  # (x, y)
            for j in range(self.grid_width)]
            for i in range(self.grid_height)])

        # intialize weight grid
        start_std = 100
        self.weights = np.zeros((self.grid_height, self.grid_width))
        self.initialize_weights_gaussian(start_pos, start_std)

    def initialize_weights_gaussian(self, start_pos, sigma):
        x0, y0 = start_pos
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                x, y = self.centers[i, j]
                dx = x - x0
                dy = y - y0
                dist_squared = dx**2 + dy**2
                self.weights[i][j] = np.exp(-dist_squared / (2 * sigma**2))
        self.weights /= np.sum(self.weights) 

    def predict(self, mu, dt):
        self.weights = predict_jit(self.weights, self.centers, mu, dt, self.res)

    def update(self, z_k, beacon_positions, sensor_noise):
        # Get full (x, y) coordinate grids
        centers_x = self.centers[:, :, 0]
        centers_y = self.centers[:, :, 1]

        # Start with all likelihoods as 1
        combined_likelihoods = np.ones_like(self.weights)

        for z_k_i, beacon_pos in zip(z_k, beacon_positions):
            dx = centers_x - beacon_pos[0]
            dy = centers_y - beacon_pos[1]
            dist = np.sqrt(dx**2 + dy**2)

            error = z_k_i - dist
            coeff = 1.0 / (np.sqrt(2 * np.pi) * sensor_noise)
            exponent = - (error ** 2) / (2 * sensor_noise ** 2)
            likelihoods = coeff * np.exp(exponent)

            combined_likelihoods *= likelihoods  # assuming independence

        self.weights *= combined_likelihoods
        total = np.sum(self.weights)
        if total > 0:
            self.weights /= total
        else:
            print("weights messed up, normalizing uniformly")
            self.weights[:] = 1.0 / self.N_s
    
    def get_grid(self):
        return self.weights
    
    def get_dimensions(self):
        return [self.grid_height, self.grid_width]
    
    def get_estimated_state(self):
        est = estimate_jit(self.weights, self.centers)
        return np.array([est[0], est[1]])
