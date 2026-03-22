import torch
import torch.nn as nn


class SimpleCNN1D(nn.Module):
    def __init__(self, in_channels: int = 12, embedding_dim: int = 128, num_classes: int = 1):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.proj = nn.Linear(128, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward_features(self, x):
        # x: (B, C, T)
        z = self.encoder(x).squeeze(-1)
        z = self.proj(z)
        return z

    def forward(self, x):
        z = self.forward_features(x)
        logits = self.classifier(z)
        return logits