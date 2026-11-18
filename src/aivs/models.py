"""Small, seeded CPU models and locally generated float32 inputs."""

import numpy as np
import torch
from torch import nn

from aivs.config import MODELS


class AttentionMatmul(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(32, 32, bias=False)
        self.k = nn.Linear(32, 32, bias=False)
        self.v = nn.Linear(32, 32, bias=False)

    def forward(self, x):
        weights = torch.softmax(self.q(x) @ self.k(x).transpose(-2, -1) / (32**0.5), -1)
        return weights @ self.v(x)


def build_model(model_id: str, seed: int) -> nn.Module:
    if model_id not in MODELS:
        raise ValueError(f"unknown model {model_id}")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if model_id == "mlp_gemm":
            model = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 64))
        elif model_id == "conv2d":
            model = nn.Sequential(
                nn.Conv2d(3, 8, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(8, 8, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(1),
            )
        else:
            model = AttentionMatmul()
    return model.cpu().eval()


def make_input(model_id: str, batch: int, seed: int, max_megabytes: float = 128):
    shape = {
        "mlp_gemm": (batch, 128),
        "conv2d": (batch, 3, 64, 64),
        "attention_matmul": (batch, 16, 32),
    }[model_id]
    if np.prod(shape) * 4 > max_megabytes * 1024**2:
        raise ValueError("input exceeds configured memory budget")
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)
