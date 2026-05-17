import os
import numpy as np

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "neural_filter_weights.pt")

# Input features per timestep: [z_k (1 beacon), mu_x, mu_y, dt]
INPUT_SIZE  = 4
HIDDEN_SIZE = 128
NUM_LAYERS  = 2

# Normalization constants shared with the training script
RANGE_SCALE = 1442.0        # screen diagonal (1200^2 + 800^2)^0.5
MU_SCALE    = 2000.0        # rough max velocity magnitude
DT_SCALE    = 1.0 / 60.0   # expected timestep

try:
    import torch
    import torch.nn as nn

    class _LSTMFilter(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, batch_first=True)
            self.head = nn.Sequential(
                nn.Linear(HIDDEN_SIZE, 64),
                nn.ReLU(),
                nn.Linear(64, 4),   # mean_x, mean_y, log_std_x, log_std_y  (normalized)
            )

        def forward(self, x, hidden=None):
            out, hidden = self.lstm(x, hidden)
            return self.head(out), hidden

    _TORCH_OK = True

except ImportError:
    _TORCH_OK = False


class NeuralFilter:
    """
    LSTM-based learned Bayesian filter.
    Trained offline via train_neural_filter.py; loads weights at init.
    Interface matches the other filters: predict(mu, dt) then update(z_k, ...).
    Outputs a Gaussian posterior (mean + std) for display alongside EKF ellipse.
    """

    def __init__(self, width, height):
        self.width  = width
        self.height = height
        self._state  = np.array([width / 2.0, height / 2.0])
        self._std    = np.array([width / 3.0, height / 3.0])
        self._mu     = np.zeros(2)
        self._dt     = DT_SCALE
        self._hidden = None
        self.ready   = False

        if not _TORCH_OK:
            print("[NeuralFilter] PyTorch not available — skipping.")
            return
        if not os.path.exists(WEIGHTS_PATH):
            print(f"[NeuralFilter] No weights found. Run train_neural_filter.py first.")
            return

        self._model = _LSTMFilter()
        self._model.load_state_dict(
            torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
        )
        self._model.eval()
        self.ready = True

    def predict(self, mu, dt):
        self._mu = np.asarray(mu, dtype=float)
        self._dt = float(dt)

    def update(self, z_k, beacon_positions, sensor_std):
        if not self.ready:
            return
        import torch
        with torch.no_grad():
            feat = [
                float(z_k[0]) / RANGE_SCALE,
                self._mu[0]   / MU_SCALE,
                self._mu[1]   / MU_SCALE,
                self._dt      / DT_SCALE,
            ]
            x = torch.tensor([[feat]], dtype=torch.float32)   # (1, 1, 4)
            params, self._hidden = self._model(x, self._hidden)
            p = params[0, 0].numpy()
            self._state = np.array([p[0] * self.width, p[1] * self.height])
            self._std   = np.array([
                np.exp(float(np.clip(p[2], -4, 2))) * self.width,
                np.exp(float(np.clip(p[3], -4, 2))) * self.height,
            ])

    def get_estimated_state(self):
        return self._state.copy()

    def get_std(self):
        return self._std.copy()
