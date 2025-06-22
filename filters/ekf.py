class EKF:
    def __init__(self, initial_state, initial_covariance):
        self.state = initial_state
        self.covariance = initial_covariance

    def predict(self, control_input, dt):
        pass

    def update(self, measurement, sensor_pos):
        pass

    def get_state(self):
        return self.state
