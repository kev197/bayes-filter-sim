import numpy as np

class GridFilter:
    def __init__(self, width, height, resolution):
        self.resolution = resolution
        self.grid_width = int(width / resolution)
        self.grid_height = int(height / resolution)
        self.belief = np.full((self.grid_width, self.grid_height), 1.0 / (self.grid_width * self.grid_height))

    def predict(self, control_input, dt):
        pass

    def update(self, measurement, beacon_pos, sensor_std=25.0):
        pass

    def estimate(self):
        total_prob = np.sum(self.belief)
        xs, ys = np.meshgrid(np.arange(self.grid_width), np.arange(self.grid_height), indexing='ij')
        x_est = np.sum(xs * self.belief) * self.resolution + self.resolution / 2
        y_est = np.sum(ys * self.belief) * self.resolution + self.resolution / 2
        return x_est / total_prob, y_est / total_prob
