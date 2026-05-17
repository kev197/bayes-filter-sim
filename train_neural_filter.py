"""
Train the LSTM-based neural filter.

Generates synthetic trajectories by running the sim physics headlessly,
then trains a 2-layer LSTM to predict position from noisy range measurements
and control inputs. Saves weights to filters/neural_filter_weights.pt.

Usage:
    py -3.12 train_neural_filter.py
"""

import os, sys, math, random
import numpy as np

# headless pygame — must be set before any pygame import
os.environ["SDL_VIDEODRIVER"] = "dummy"
import pygame
pygame.init()
pygame.display.set_mode((1200, 800))

sys.path.insert(0, os.path.dirname(__file__))

from sim.world import World
from sim.agent import Agent
from filters.neural_filter import (
    _LSTMFilter, WEIGHTS_PATH,
    INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS,
    RANGE_SCALE, MU_SCALE, DT_SCALE,
)

import torch
import torch.nn as nn

WIDTH       = 1200
HEIGHT      = 800
NUM_BEACONS = 1
SENSOR_NOISE = 30.0
DT          = 1.0 / 60.0


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def collect_trajectory(T: int, seed: int):
    random.seed(seed)
    np.random.seed(seed)

    world = World(WIDTH, HEIGHT, NUM_BEACONS)
    agent = Agent(start_x=700, start_y=250, frame_rate=60,
                  WIDTH=WIDTH, HEIGHT=HEIGHT, radius=10, speed=11)

    speed = random.gauss(550, 80)
    angle = random.uniform(0, 2 * math.pi)
    mu    = np.array([speed * math.cos(angle), speed * math.sin(angle)])

    fake_keys = {}   # non-manual branch of agent.move() never reads keys

    obs_seq, ctrl_seq, pos_seq = [], [], []

    for _ in range(T):
        mu, _ = agent.move(fake_keys, world, mu, DT, manual_control=False)
        true_pos = agent.get_position().astype(float)

        deltas    = world.beacons - true_pos
        distances = np.linalg.norm(deltas, axis=1)
        z_k       = distances + np.random.normal(0, SENSOR_NOISE, size=NUM_BEACONS)

        obs_seq.append(z_k.copy())
        ctrl_seq.append(np.array([mu[0], mu[1], DT]))
        pos_seq.append(true_pos.copy())

    return np.array(obs_seq), np.array(ctrl_seq), np.array(pos_seq)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(n_episodes: int = 200, T: int = 400, epochs: int = 150):
    print(f"Generating {n_episodes} trajectories × {T} steps...")
    all_obs, all_ctrl, all_pos = [], [], []
    for i in range(n_episodes):
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n_episodes}")
        obs, ctrl, pos = collect_trajectory(T=T, seed=i)
        all_obs.append(obs)
        all_ctrl.append(ctrl)
        all_pos.append(pos)

    # tensors: (N, T, ?)
    obs_t  = torch.tensor(np.array(all_obs),  dtype=torch.float32)
    ctrl_t = torch.tensor(np.array(all_ctrl), dtype=torch.float32)
    pos_t  = torch.tensor(np.array(all_pos),  dtype=torch.float32)

    # normalize
    obs_t          = obs_t  / RANGE_SCALE
    ctrl_t[..., 0] = ctrl_t[..., 0] / MU_SCALE
    ctrl_t[..., 1] = ctrl_t[..., 1] / MU_SCALE
    ctrl_t[..., 2] = ctrl_t[..., 2] / DT_SCALE   # all 1.0 (fixed DT)
    pos_t[..., 0]  = pos_t[..., 0]  / WIDTH
    pos_t[..., 1]  = pos_t[..., 1]  / HEIGHT

    x_in = torch.cat([obs_t, ctrl_t], dim=-1)   # (N, T, 4)

    model     = _LSTMFilter()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    WARMUP = 15   # skip first steps while hidden state initialises

    print(f"Training LSTM for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        params, _ = model(x_in)                       # (N, T, 4)
        mean    = params[..., :2]                     # (N, T, 2)
        log_std = params[..., 2:].clamp(-4, 2)
        std     = torch.exp(log_std)

        nll = (0.5 * ((pos_t[:, WARMUP:] - mean[:, WARMUP:]) / std[:, WARMUP:]) ** 2
               + log_std[:, WARMUP:]).mean()

        nll.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 30 == 0:
            with torch.no_grad():
                rmse_x = (((pos_t[:, WARMUP:, 0] - mean[:, WARMUP:, 0]) * WIDTH)  ** 2).mean().sqrt()
                rmse_y = (((pos_t[:, WARMUP:, 1] - mean[:, WARMUP:, 1]) * HEIGHT) ** 2).mean().sqrt()
            print(f"  epoch {epoch+1:3d}/{epochs}  NLL={nll.item():.4f}  "
                  f"RMSE x={rmse_x:.1f}px  y={rmse_y:.1f}px")

    torch.save(model.state_dict(), WEIGHTS_PATH)
    print(f"\nWeights saved to {WEIGHTS_PATH}")


if __name__ == "__main__":
    train()
