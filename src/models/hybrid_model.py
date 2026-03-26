import torch
import torch.nn as nn
from cnn_model import SimpleCNN1D


class HybridCNN(nn.Module):
    def __init__(self, in_channels: int = 12, handcrafted_dim: int = 16, embedding_dim: int = 128):
        super().__init__()

        self.cnn = SimpleCNN1D(in_channels=in_channels, embedding_dim=embedding_dim, num_classes=1)

        self.handcrafted_branch = nn.Sequential(
            nn.Linear(handcrafted_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim + 64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x_signal, x_feat):
        z_signal = self.cnn.forward_features(x_signal)
        z_feat = self.handcrafted_branch(x_feat)
        z = torch.cat([z_signal, z_feat], dim=1)
        logits = self.classifier(z)
        return logits