import math
import random
import numpy as np

class EKF:
    def __init__(self, initial_state, initial_covariance, beacon_pos, sensor_noise):
        self.state = initial_state
        self.covariance = initial_covariance
        self.beacon_pos = beacon_pos
        self.sensor_noise = sensor_noise

    def predict(self, mu, dt):
        # x = x_0 + v_x*dt + noise
        # y = y_0 + v_y*dt + noise
        mu = mu * 0.8
        self.state[0] += mu[0] * dt
        self.state[1] += mu[1] * dt

        # propagate covariance matrix through uncertainty of prior 
        # Q + FPF^T
        a = 7
        b = 2
        scale_process_noise = 5.0
        var_alpha = (a * b) / ((a + b) ** 2 * (a + b + 1))
        q_x = var_alpha * (mu[0] * dt)**2
        q_y = var_alpha * (mu[1] * dt)**2
        Q = scale_process_noise * np.array([[q_x, 0],
                     [0, q_y]])
        F = np.eye(2)
        self.covariance = Q + F @ self.covariance @ F.T

    def update(self, z_k):
        # since the observation model is nonlinear in this case (square root)
        # we must take the jacobian of the observation model evaluated at the 
        # predicted state. this gives us the first order taylor approximation 
        # of the observation locally, which is a linear function
        # that can be propagated through the normal KF update.
        dx = self.state[0] - self.beacon_pos[0]
        dy = self.state[1] - self.beacon_pos[1]
        dist = np.sqrt(dx**2 + dy**2)
        H = None
        if dist == 0:
            H = np.zeroes((1, 2))
        else:
            H = np.array([[dx / dist, dy / dist]])
        R = np.array([[self.sensor_noise]])
        S = H @ self.covariance @ H.T + R
        K = self.covariance @ H.T @ np.linalg.inv(S)
        self.state = self.state + K @ (z_k - np.array([dist]))

        # bayesian fusion principles tells us that the generated gaussian has 
        # less covariance than the original gaussians (the prior and likelihood). 
        # However, we must scale this by the kalman control, which is how much of 
        # a shift we actually performed towards the peak overlap of these gaussians. 
        self.covariance = self.covariance - K @ H @ self.covariance

    def get_state(self):
        return self.state
    
    def get_covariance(self):
        return self.covariance
