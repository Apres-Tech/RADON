"""Custom feature extractor for RADON's Dict observation space."""

import gymnasium.spaces as spaces
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class RescueExtractor(BaseFeaturesExtractor):
    """Custom features extractor tailored for 6x7x7 local grid and 6-dim self vector.

    Architecture:
      - grid branch: Conv2d(6->32, k=3, pad=1) -> ReLU -> Conv2d(32->64, k=3, pad=1) -> ReLU
                     -> Flatten -> Linear(64*7*7 -> 128) -> ReLU
      - self branch: Linear(6 -> 32) -> ReLU
      - concatenation: 128 + 32 = 160 features
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 160):
        super().__init__(observation_space, features_dim=features_dim)

        # 6x7x7 grid CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
        )

        # 6-dim self vector MLP
        self.mlp = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        """Extracts and concatenates representations from grid and self vector."""
        grid_features = self.cnn(observations["grid"])
        self_features = self.mlp(observations["self"])
        return torch.cat([grid_features, self_features], dim=1)

