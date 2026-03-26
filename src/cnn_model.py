"""
src/model.py

1D-CNN for Brugada syndrome ECG classification.

Architecture:
  4 convolutional blocks (Conv1d → BN → ReLU → Pool)
  + FC head with 64-dim penultimate embedding (EXPOSED for downstream fusion)
  + binary output head (logits during training, sigmoid at inference)

Design constraints:
  - ~220K parameters (3-lead) — appropriate for 363-subject dataset
  - Embedding dimension = 64 (contract with Person 4)
  - Dropout 0.5 for regularization
  - Logits output — use BCEWithLogitsLoss during training
"""

from typing import Tuple

import torch
import torch.nn as nn


# =============================================================================
# CONVOLUTIONAL BLOCK
# =============================================================================

class ConvBlock1D(nn.Module):
    """
    Residual-free convolutional block:
      Conv1d → BatchNorm1d → ReLU → Pool
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        pool_size: int = 2,
        use_maxpool: bool = True,
    ):
        super().__init__()
        padding = kernel_size // 2   # "same" padding (approximately)

        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,   # BN subsumes bias
        )
        self.bn   = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if use_maxpool:
            self.pool = nn.MaxPool1d(pool_size)
        else:
            self.pool = nn.AdaptiveAvgPool1d(1)   # used in Block 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


# =============================================================================
# MAIN MODEL
# =============================================================================

class BrugadaCNN(nn.Module):
    """
    1D-CNN for Brugada ECG binary classification.

    Args:
        in_channels:  number of ECG leads (3 for V1/V2/V3, 12 for full)
        embed_dim:    dimension of penultimate embedding (default 64 — contract)
        dropout:      dropout rate in FC head
        n_timesteps:  used only for parameter counting / shape notes

    Forward:
        Input:  (batch, in_channels, 1200)
        Output: (batch, 1)  — logits (use sigmoid for probability)

    To get embeddings:
        Call model.get_embedding(x) — returns (batch, embed_dim)
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 64,
        dropout: float = 0.5,
        n_timesteps: int = 1200,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.embed_dim   = embed_dim

        # ------------------------------------------------------------------
        # CONVOLUTIONAL BACKBONE
        # Block 1: kernel=7, pool → /2
        # Block 2: kernel=5, pool → /2
        # Block 3: kernel=5, pool → /2
        # Block 4: kernel=3, AdaptiveAvgPool → (batch, 256, 1)
        # ------------------------------------------------------------------
        self.block1 = ConvBlock1D(in_channels,  32,  kernel_size=7, pool_size=2, use_maxpool=True)
        self.block2 = ConvBlock1D(32,           64,  kernel_size=5, pool_size=2, use_maxpool=True)
        self.block3 = ConvBlock1D(64,           128, kernel_size=5, pool_size=2, use_maxpool=True)
        self.block4 = ConvBlock1D(128,          256, kernel_size=3, use_maxpool=False)   # AdaptiveAvgPool

        # ------------------------------------------------------------------
        # FULLY CONNECTED HEAD
        # ------------------------------------------------------------------
        self.flatten = nn.Flatten()

        # Penultimate embedding layer (64-dim) — MUST be exposed
        self.fc_embed = nn.Sequential(
            nn.Linear(256, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

        # Output head: logits only
        self.fc_out = nn.Linear(embed_dim, 1)

        # ------------------------------------------------------------------
        # WEIGHT INITIALIZATION
        # ------------------------------------------------------------------
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_channels, T)
        Returns:
            logits: (batch, 1)
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.flatten(x)         # (batch, 256)
        x = self.fc_embed(x)        # (batch, embed_dim)  ← penultimate
        x = self.fc_out(x)          # (batch, 1)  ← logits
        return x

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract penultimate-layer embedding without running the output head.

        Args:
            x: (batch, in_channels, T)
        Returns:
            embedding: (batch, embed_dim)

        USAGE: call in eval mode with torch.no_grad() for OOF extraction.
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.flatten(x)
        x = self.fc_embed(x)    # (batch, embed_dim) — output of ReLU, before fc_out
        return x

    def forward_with_embedding(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return both logits and embedding in a single forward pass.
        More efficient than calling forward + get_embedding separately.

        Returns:
            logits:    (batch, 1)
            embedding: (batch, embed_dim)
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.flatten(x)
        embed  = self.fc_embed(x)   # (batch, embed_dim)
        logits = self.fc_out(embed) # (batch, 1)
        return logits, embed

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# FACTORY
# =============================================================================

def build_model(in_channels: int = 3, embed_dim: int = 64,
                dropout: float = 0.5) -> BrugadaCNN:
    model = BrugadaCNN(in_channels=in_channels, embed_dim=embed_dim, dropout=dropout)
    return model