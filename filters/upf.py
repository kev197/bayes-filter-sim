import math, random
import numpy as np

# Unscented Particle Filter (UPF)
# van der Merwe et al. (2000); discussed in Arulampalam et al. (2002) §IV-C.
#
# Core idea: instead of sampling from the blind prior p(x_k | x_{k-1}^i) like SIR,
# each particle runs a mini UKF step to build a measurement-informed Gaussian proposal
# q(x_k | x_{k-1}^i, z_k) = N(μ_k^i, Σ_k^i). With a range-only beacon, the UKF
# update concentrates sigma points near the measurement circle before drawing the new
# sample — that's where SIR wastes most of its particles.
#
# Weight: w^i ∝ w_{k-1}^i * p(z_k | x_k^i)
# (the full prior/proposal ratio is omitted; numerically unstable with non-Gaussian
# dynamics and unbounded tangential covariance from a single bearing-free beacon)

_N     = 2      # state dimension [x, y]
_ALPHA = 0.3
_BETA  = 2.0
_KAPPA = 0.0
_LAM   = _ALPHA**2 * (_N + _KAPPA) - _N

_Wm = np.array([_LAM / (_N + _LAM)] + [1 / (2 * (_N + _LAM))] * (2 * _N))
_Wc = np.array([_LAM / (_N + _LAM) + (1 - _ALPHA**2 + _BETA)]
               + [1 / (2 * (_N + _LAM))] * (2 * _N))


class Particle:
    def __init__(self, x, y, weight=1.0):
        self.x = x
        self.y = y
        self.weight = weight


class UnscentedParticleFilter:
    def __init__(self, N_s, width, height):
        self.N_s = N_s

        # Fixed process noise covariance — used for every particle's UKF sigma points.
        # Sized to match the dominant OU + dynamics noise in the sim (~35 px/step).
        q = 35.0 ** 2
        self.Q = np.diag([q, q])

        deviation = 200
        self.particles = [
            Particle(
                random.uniform(700 - deviation, 700 + deviation),
                random.uniform(250 - deviation, 250 + deviation),
                weight=1.0 / N_s,
            )
            for _ in range(N_s)
        ]
        self._mu = np.zeros(2)
        self._dt = 0.0

    # ------------------------------------------------------------------
    def predict(self, mu, dt):
        self._mu = np.asarray(mu, dtype=float)
        self._dt = float(dt)

    # ------------------------------------------------------------------
    # UKF internals
    # ------------------------------------------------------------------

    def _sigma_points(self, x, P):
        try:
            L = np.linalg.cholesky((_N + _LAM) * P)
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky((_N + _LAM) * P + 1e-5 * np.eye(_N))
        return [x] + [x + L[:, i] for i in range(_N)] + [x - L[:, i] for i in range(_N)]

    def _f(self, x):
        """Deterministic mean dynamics: E[Beta(6,2)] = 0.75."""
        return x + self._mu * 0.75 * self._dt

    @staticmethod
    def _h(x, beacon_positions):
        return np.array([math.sqrt((x[0] - bx)**2 + (x[1] - by)**2)
                         for bx, by in beacon_positions])

    def _ukf_proposal(self, x_prev, z_k, beacon_positions, sensor_std):
        """Return (μ_prop, P_prop) — the UKF posterior used as the proposal."""
        m = len(z_k)
        R = np.eye(m) * sensor_std**2

        # Predict
        sigma  = self._sigma_points(x_prev, self.Q)
        sp     = [self._f(s) for s in sigma]
        x_pred = sum(w * s for w, s in zip(_Wm, sp))
        P_pred = self.Q.copy()
        for w, s in zip(_Wc, sp):
            d = s - x_pred
            P_pred += w * np.outer(d, d)

        # Update
        sigma2 = self._sigma_points(x_pred, P_pred)
        z_pts  = [self._h(s, beacon_positions) for s in sigma2]
        z_hat  = sum(w * zp for w, zp in zip(_Wm, z_pts))
        S      = R.copy()
        Pxz    = np.zeros((_N, m))
        for w, s, zp in zip(_Wc, sigma2, z_pts):
            dz = zp - z_hat
            dx = s  - x_pred
            S   += w * np.outer(dz, dz)
            Pxz += w * np.outer(dx, dz)

        # 1-beacon fast path: S is 1×1 — avoid full matrix inverse
        S_inv   = np.array([[1.0 / S[0, 0]]]) if m == 1 else np.linalg.inv(S)
        K       = Pxz @ S_inv
        mu_prop = x_pred + K @ (z_k - z_hat)
        P_prop  = P_pred - K @ S @ K.T
        P_prop  = (P_prop + P_prop.T) / 2 + 1e-3 * np.eye(_N)

        return mu_prop, P_prop

    def _log_likelihood(self, x, z_k, beacon_positions, sensor_std):
        ll = 0.0
        for z_i, (bx, by) in zip(z_k, beacon_positions):
            dist = math.sqrt((x[0] - bx)**2 + (x[1] - by)**2)
            ll += (-0.5 * ((z_i - dist) / sensor_std)**2
                   - math.log(math.sqrt(2 * math.pi) * sensor_std))
        return ll

    # ------------------------------------------------------------------
    # UPF step
    # ------------------------------------------------------------------

    def update(self, z_k, beacon_positions, sensor_std):
        new_particles = []
        log_weights   = []

        for p in self.particles:
            x_prev = np.array([p.x, p.y])

            mu_prop, P_prop = self._ukf_proposal(x_prev, z_k, beacon_positions, sensor_std)

            try:
                x_new = np.random.multivariate_normal(mu_prop, P_prop)
            except Exception:
                x_new = mu_prop.copy()

            log_w = (math.log(p.weight + 1e-300)
                     + self._log_likelihood(x_new, z_k, beacon_positions, sensor_std))

            new_particles.append(Particle(x_new[0], x_new[1]))
            log_weights.append(log_w)

        log_weights = np.array(log_weights)
        log_weights -= log_weights.max()
        weights = np.exp(log_weights)
        weights /= weights.sum()
        for p, w in zip(new_particles, weights):
            p.weight = float(w)

        self.particles = new_particles

    def resample(self):
        weights = np.array([p.weight for p in self.particles])
        cdf = np.cumsum(weights)
        cdf[-1] = 1.0
        u1 = random.uniform(0, 1.0 / self.N_s)
        indices, i = [], 0
        for j in range(self.N_s):
            uj = u1 + j / self.N_s
            while i < self.N_s - 1 and uj > cdf[i]:
                i += 1
            indices.append(i)
        self.particles = [
            Particle(self.particles[k].x, self.particles[k].y, 1.0 / self.N_s)
            for k in indices
        ]

    def effective_sample_size(self):
        s = sum(p.weight**2 for p in self.particles)
        return 1.0 / s if s > 0 else float(self.N_s)

    def get_estimated_state(self):
        x = sum(p.x * p.weight for p in self.particles)
        y = sum(p.y * p.weight for p in self.particles)
        return np.array([x, y])
