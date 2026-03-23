"""
model.py — 1D-CNN architecture for Brugada syndrome classification.

Two model variants:
  - BrugadaCNN1D:   simple 4-block 1D-CNN (recommended baseline)
  - BrugadaResNet1D: residual 1D-CNN (optional, Phase 5+)

Both models expose a 64-dim penultimate embedding vector for downstream fusion.

Input shape:  (batch, n_leads, n_timesteps)
Output:
  - logits:    (batch, 1) — use BCEWithLogitsLoss during training
  - embedding: (batch, 64) — exposed via forward(x, return_embedding=True)
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import DROPOUT, EMBEDDING_DIM

logger = logging.getLogger(__name__)


# ============================================================
# CONV BLOCK
# ============================================================

class ConvBlock1D(nn.Module):
    """Standard Conv1D -> BN -> ReLU -> Pool block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        pool_size: Optional[int] = 2,
        pool_type: str = "max",   # "max" or "avg" or None
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=kernel_size // 2   # same-ish padding
        )
        self.bn   = nn.BatchNorm1d(out_channels)
        self.act  = nn.ReLU(inplace=True)

        if pool_type == "max" and pool_size:
            self.pool = nn.MaxPool1d(pool_size)
        elif pool_type == "avg" and pool_size:
            self.pool = nn.AvgPool1d(pool_size)
        else:
            self.pool = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.act(self.bn(self.conv(x))))


# ============================================================
# SIMPLE 1D-CNN (RECOMMENDED BASELINE)
# ============================================================

class BrugadaCNN1D(nn.Module):
    """
    Simple 4-block 1D-CNN for Brugada ECG classification.

    Architecture (from Work Package B):
        Block 1: Conv(leads->32, k=7) -> BN -> ReLU -> MaxPool(2)
        Block 2: Conv(32->64, k=5)    -> BN -> ReLU -> MaxPool(2)
        Block 3: Conv(64->128, k=5)   -> BN -> ReLU -> MaxPool(2)
        Block 4: Conv(128->256, k=3)  -> BN -> ReLU -> AdaptiveAvgPool(1)
        Head:    Flatten -> Linear(256->64) -> ReLU -> Dropout(0.5)
        Output:  Linear(64->1) [logits]

    Args:
        n_leads: number of input leads (3 for V1-V3 model, 12 for full)
        embedding_dim: penultimate layer size (default 64)
        dropout: dropout probability in head
    """

    def __init__(
        self,
        n_leads:       int   = 12,
        embedding_dim: int   = EMBEDDING_DIM,
        dropout:       float = DROPOUT,
    ):
        super().__init__()
        self.n_leads       = n_leads
        self.embedding_dim = embedding_dim

        # ---- Convolutional backbone ----
        self.block1 = ConvBlock1D(n_leads, 32,  kernel_size=7, pool_size=2, pool_type="max")
        self.block2 = ConvBlock1D(32,      64,  kernel_size=5, pool_size=2, pool_type="max")
        self.block3 = ConvBlock1D(64,      128, kernel_size=5, pool_size=2, pool_type="max")
        self.block4 = ConvBlock1D(128,     256, kernel_size=3, pool_size=None)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # ---- Classification head ----
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

        # ---- Output ----
        self.classifier = nn.Linear(embedding_dim, 1)

        # ---- Weight initialisation ----
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (batch, n_leads, timesteps)
            return_embedding: if True, also return the 64-dim embedding

        Returns:
            logits: (batch, 1)
            embedding: (batch, embedding_dim) or None
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_pool(x)

        embedding = self.head(x)            # (batch, embedding_dim)
        logits    = self.classifier(embedding)  # (batch, 1)

        if return_embedding:
            return logits, embedding
        return logits, None

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
# RESIDUAL BLOCK (for ResNet-1D variant)
# ============================================================

class ResBlock1D(nn.Module):
    """
    1D Residual block with optional downsampling.
    Designed for Phase 5 (optional upgrade from simple CNN).
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        kernel_size:  int = 3,
        stride:       int = 1,
        dropout:      float = 0.0,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=kernel_size // 2, bias=False
        )
        self.bn1    = nn.BatchNorm1d(out_channels)
        self.conv2  = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            stride=1, padding=kernel_size // 2, bias=False
        )
        self.bn2    = nn.BatchNorm1d(out_channels)
        self.drop   = nn.Dropout(p=dropout)

        # Skip connection — adjust channels / length if needed
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return F.relu(out + identity, inplace=True)


class BrugadaResNet1D(nn.Module):
    """
    Residual 1D-CNN for Brugada classification.
    Use only if simple CNN is stable (Phase 5).

    Architecture:
        Stem: Conv(n_leads->64, k=7, s=2) -> BN -> ReLU
        Layer1: 2× ResBlock(64,  64)
        Layer2: 2× ResBlock(64,  128, stride=2)
        Layer3: 2× ResBlock(128, 256, stride=2)
        GlobalAvgPool -> Linear(256->embedding_dim) -> Dropout -> Linear(->1)
    """

    def __init__(
        self,
        n_leads:       int   = 12,
        embedding_dim: int   = EMBEDDING_DIM,
        dropout:       float = DROPOUT,
    ):
        super().__init__()
        self.n_leads       = n_leads
        self.embedding_dim = embedding_dim

        self.stem = nn.Sequential(
            nn.Conv1d(n_leads, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        self.layer1 = nn.Sequential(ResBlock1D(64,  64),  ResBlock1D(64,  64))
        self.layer2 = nn.Sequential(ResBlock1D(64,  128, stride=2), ResBlock1D(128, 128))
        self.layer3 = nn.Sequential(ResBlock1D(128, 256, stride=2), ResBlock1D(256, 256))

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )
        self.classifier = nn.Linear(embedding_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.global_pool(x)
        embedding = self.head(x)
        logits    = self.classifier(embedding)
        if return_embedding:
            return logits, embedding
        return logits, None

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
# FACTORY
# ============================================================

def build_model(
    model_type:    str   = "cnn",     # "cnn" or "resnet"
    n_leads:       int   = 12,
    embedding_dim: int   = EMBEDDING_DIM,
    dropout:       float = DROPOUT,
) -> nn.Module:
    """
    Factory function — returns the selected model.

    Args:
        model_type: "cnn" for BrugadaCNN1D, "resnet" for BrugadaResNet1D
        n_leads: 3 (V1-V3) or 12 (full)
        embedding_dim: output embedding size
        dropout: dropout in head

    Returns:
        Instantiated (untrained) model.
    """
    if model_type == "cnn":
        model = BrugadaCNN1D(n_leads=n_leads, embedding_dim=embedding_dim, dropout=dropout)
    elif model_type == "resnet":
        model = BrugadaResNet1D(n_leads=n_leads, embedding_dim=embedding_dim, dropout=dropout)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose 'cnn' or 'resnet'.")

    n_params = model.count_parameters()
    logger.info(
        f"Built {model.__class__.__name__} | "
        f"n_leads={n_leads} | embedding_dim={embedding_dim} | "
        f"params={n_params:,}"
    )
    return model


# ============================================================
# SANITY CHECK
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import torch

    for n_leads, variant in [(3, "3lead"), (12, "12lead")]:
        for model_type in ["cnn", "resnet"]:
            model = build_model(model_type=model_type, n_leads=n_leads)

            # Test with beat-level input
            beat_x = torch.randn(8, n_leads, 70)     # (batch, leads, beat_window)
            logits, emb = model(beat_x, return_embedding=True)
            print(
                f"{model.__class__.__name__} [{variant}] | "
                f"input {tuple(beat_x.shape)} -> "
                f"logits {tuple(logits.shape)}, "
                f"emb {tuple(emb.shape)}"
            )

            # Test with full-recording input
            rec_x = torch.randn(4, n_leads, 1200)
            logits2, emb2 = model(rec_x, return_embedding=True)
            print(
                f"  Full recording: input {tuple(rec_x.shape)} -> "
                f"logits {tuple(logits2.shape)}, emb {tuple(emb2.shape)}"
            )