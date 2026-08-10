"""
Lightweight bottleneck adapter (MLP) as described in AD-DINOv3 paper.
Each adapter: linear_down -> LeakyReLU -> linear_up, with residual connection.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LightAdapter(nn.Module):
    """
    Bottleneck adapter: d -> d//r -> d, with LeakyReLU.
    Applied to visual patch tokens, CLS token, and text embeddings separately.
    """
    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        self.dim = dim
        self.reduction = reduction
        hidden_dim = max(dim // reduction, 1)
        self.down = nn.Linear(dim, hidden_dim)
        self.relu = nn.LeakyReLU(inplace=True)
        self.up = nn.Linear(hidden_dim, dim)
        # Initialize near identity for stability
        nn.init.xavier_uniform_(self.down.weight, gain=1.0)
        nn.init.zeros_(self.down.bias)
        nn.init.xavier_uniform_(self.up.weight, gain=1.0)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        residual = x
        x = self.down(x)
        x = self.relu(x)
        x = self.up(x)
        return x + residual  # light residual for calibration


class MultiLevelVisualAdapter(nn.Module):
    """
    Multi-level adaptation for DINOv3 features extracted from layers 6, 12, 18, 24.
    Each level gets its own adapter to preserve complementary cues.
    """
    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        # Four stages as per paper: layer 6, 12, 18, 24
        self.adapters = nn.ModuleList([LightAdapter(dim, reduction) for _ in range(4)])

    def forward(self, features: list) -> list:
        # features: list of 4 tensors, each (..., dim)
        adapted = []
        for adapter, feat in zip(self.adapters, features):
            adapted.append(adapter(feat))
        return adapted
