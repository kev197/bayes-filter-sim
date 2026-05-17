import math, random
import numpy as np

# Auxiliary Sampling Importance Resampling (ASIR)
# Arulampalam et al. (2002), "A Tutorial on Particle Filters for Online
# Nonlinear/Non-Gaussian Bayesian Tracking," IEEE Trans. Signal Process., Algorithm 4.
#
# Key difference from bootstrap SIR: before propagating particles, a first-stage
# resampling is done using w^i * p(z_k | mu_k^i), where mu_k^i is the predicted mean
# for particle i (not a stochastic sample). This concentrates particles in regions
# the upcoming measurement favors. The final weights correct for this pre-selection
# via the ratio p(z_k | x_k^i) / p(z_k | mu_k^{j^i}).

class Particle:
    def __init__(self, x, y, weight=1.0):
        self.x = x
        self.y = y
        self.weight = weight


class ASIRFilter:
    def __init__(self, N_s, width, height):
        self.N_s = N_s
        deviation = 200
        self.particles = [
            Particle(
                random.uniform(700 - deviation, 700 + deviation),
                random.uniform(250 - deviation, 250 + deviation),
                weight=1.0 / N_s,
            )
            for _ in range(N_s)
        ]
        self._mu = None
        self._dt = None

    def predict(self, mu, dt):
        self._mu = mu
        self._dt = dt

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log_likelihood(self, x, y, z_k, beacon_positions, sensor_std):
        ll = 0.0
        for z_k_i, beacon_pos in zip(z_k, beacon_positions):
            dx = x - beacon_pos[0]
            dy = y - beacon_pos[1]
            dist = math.sqrt(dx ** 2 + dy ** 2)
            error = z_k_i - dist
            ll += -(error ** 2) / (2 * sensor_std ** 2) - math.log(math.sqrt(2 * math.pi) * sensor_std)
        return ll

    def _predicted_mean(self, p):
        # E[Beta(6,2)] = 6/8 = 0.75, matching the motion model used in predict
        alpha = 0.75
        return (
            p.x + self._mu[0] * alpha * self._dt,
            p.y + self._mu[1] * alpha * self._dt,
        )

    def _propagate_sample(self, p):
        alpha = random.betavariate(6, 2)
        angle = random.uniform(0, 2 * math.pi)
        r = random.gauss(0.0, 8.0)
        return (
            p.x + self._mu[0] * alpha * self._dt + r * math.cos(angle),
            p.y + self._mu[1] * alpha * self._dt + r * math.sin(angle),
        )

    def _systematic_resample(self, weights):
        N_s = self.N_s
        cdf = np.cumsum(weights)
        cdf[-1] = 1.0  # guard against floating-point shortfall
        u1 = random.uniform(0, 1.0 / N_s)
        indices = []
        i = 0
        for j in range(N_s):
            uj = u1 + j / N_s
            while i < N_s - 1 and uj > cdf[i]:
                i += 1
            indices.append(i)
        return indices

    # ------------------------------------------------------------------
    # ASIR update (Algorithm 4)
    # ------------------------------------------------------------------

    def update(self, z_k, beacon_positions, sensor_std):
        N_s = self.N_s
        particles = self.particles

        # Step 1: representative points and first-stage log-weights
        #   lambda^i  ∝  w^{i}_{k-1} * p(z_k | mu_k^i)
        pred_means = [self._predicted_mean(p) for p in particles]
        log_lambdas = np.array([
            math.log(p.weight + 1e-300) + self._log_likelihood(mx, my, z_k, beacon_positions, sensor_std)
            for p, (mx, my) in zip(particles, pred_means)
        ])

        # Subtract max before exponentiating for numerical stability
        log_lambdas -= log_lambdas.max()
        lambdas = np.exp(log_lambdas)
        lambdas /= lambdas.sum()

        # Step 2: systematic resample to select N_s ancestor indices
        indices = self._systematic_resample(lambdas)

        # Step 3 & 4: propagate resampled ancestors; compute correction weights
        #   w_k^i  ∝  p(z_k | x_k^i) / p(z_k | mu_k^{j^i})
        new_particles = []
        log_weights = []
        for j in indices:
            src = particles[j]
            mx, my = pred_means[j]

            new_x, new_y = self._propagate_sample(src)

            log_w = (
                self._log_likelihood(new_x, new_y, z_k, beacon_positions, sensor_std)
                - self._log_likelihood(mx, my, z_k, beacon_positions, sensor_std)
            )
            new_particles.append(Particle(new_x, new_y))
            log_weights.append(log_w)

        # Normalize in log-space
        log_weights = np.array(log_weights)
        log_weights -= log_weights.max()
        weights = np.exp(log_weights)
        weights /= weights.sum()

        for p, w in zip(new_particles, weights):
            p.weight = float(w)

        self.particles = new_particles

    def effective_sample_size(self):
        s = sum(p.weight ** 2 for p in self.particles)
        return 1.0 / s if s > 0 else float(self.N_s)

    def get_estimated_state(self):
        x = sum(p.x * p.weight for p in self.particles)
        y = sum(p.y * p.weight for p in self.particles)
        return np.array([x, y])
