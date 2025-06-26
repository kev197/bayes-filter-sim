import math
import random
import numpy as np

class EKF:
    def __init__(self, initial_state, initial_covariance, beacon_positions, sensor_noise):
        self.state = initial_state
        self.covariance = initial_covariance
        self.beacon_positions = beacon_positions
        self.sensor_noise = sensor_noise

    def predict(self, mu, dt):
        # x = x_0 + v_x*dt + noise
        # y = y_0 + v_y*dt + noise
        alpha = 0.75
        self.state[0] += mu[0] * dt * alpha
        self.state[1] += mu[1] * dt * alpha

        # propagate covariance matrix through uncertainty of prior 
        # Q + FPF^T

        # (following alpha scaling of velocity, variance of the beta)
        # a = 50
        # b = 2
        # scale_process_noise = 5.0
        # var_alpha = (a * b) / ((a + b) ** 2 * (a + b + 1))
        # q_x = var_alpha * (mu[0] * dt)**2
        # q_y = var_alpha * (mu[1] * dt)**2
        # Q = scale_process_noise * np.array([[q_x, 0],
        #              [0, q_y]])

        process_std = 4.0  # adjust based on expected actuator noise
        q = process_std ** 2
        Q = np.array([[q, 0],
                    [0, q]])
        F = np.eye(2)
        self.covariance = Q + F @ self.covariance @ F.T

    def h(self, x, beacon_positions):
        return np.array([
            np.sqrt((x[0] - bx)**2 + (x[1] - by)**2)
            for (bx, by) in beacon_positions
        ])
    
    def compute_jacobian(self, x, beacon_positions):
        H = []
        for (bx, by) in beacon_positions:
            dx = x[0] - bx
            dy = x[1] - by
            dist = np.sqrt(dx**2 + dy**2)
            if dist == 0:
                H.append([0, 0])
            else:
                H.append([dx / dist, dy / dist])
        return np.array(H)  # Shape: (num_beacons, 2)

    def update(self, z_k):
        # since the observation model is nonlinear in this case (square root)
        # we must take the jacobian of the observation model evaluated at the 
        # predicted state. this gives us the first order taylor approximation 
        # of the observation locally, which is a linear function
        # that can be propagated through the normal KF update.
        z_hat_k = self.h(self.state, self.beacon_positions)
        H = self.compute_jacobian(self.state, self.beacon_positions)

        R = np.eye(len(self.beacon_positions)) * self.sensor_noise**2

        S = H @ self.covariance @ H.T + R
        K = self.covariance @ H.T @ np.linalg.inv(S)
        self.state = self.state + K @ (z_k - z_hat_k)

        # bayesian fusion principles tells us that the generated gaussian has 
        # less covariance than the original gaussians (the prior and likelihood). 
        # However, we must scale this by the kalman control, which is how much of 
        # a shift we actually performed towards the peak overlap of these gaussians. 
        self.covariance = self.covariance - K @ H @ self.covariance

    def get_state(self):
        return self.state
    
    def get_covariance(self):
        return self.covariance
