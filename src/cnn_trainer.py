"""
src/trainer.py

Training engine for the Brugada 1D-CNN.

Features:
  - BCEWithLogitsLoss with pos_weight for class imbalance
  - Early stopping on validation AUROC (primary metric)
  - Best model checkpoint saving per fold
  - OOF embedding and probability extraction
  - Full CPU compatibility (no CUDA dependency)
"""

import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from sklearn.metrics import roc_auc_score

from src.cnn_utils import ensure_dirs, get_logger

logger = get_logger("trainer")


# =============================================================================
# EARLY STOPPING
# =============================================================================

class EarlyStopping:
    """
    Stop training when validation AUROC has not improved for `patience` epochs.
    Saves the best model state dict.
    """

    def __init__(self, patience: int = 15, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_score = -np.inf
        self.counter    = 0
        self.best_state: Optional[Dict] = None
        self.best_epoch = 0

    def step(self, score: float, model: nn.Module, epoch: int) -> bool:
        """
        Args:
            score: validation AUROC (higher = better)
            model: current model
            epoch: current epoch number

        Returns:
            True if training should stop.
        """
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter    = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.best_epoch = epoch
        else:
            self.counter += 1

        return self.counter >= self.patience

    def restore_best(self, model: nn.Module) -> nn.Module:
        """Load the best checkpoint back into the model."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
            logger.info(f"Restored best model from epoch {self.best_epoch} "
                        f"(val AUROC={self.best_score:.4f})")
        return model


# =============================================================================
# ONE EPOCH
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run one training epoch. Returns mean loss."""
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for signals, labels in loader:
        signals = signals.to(device)            # (batch, C, T)
        labels  = labels.to(device).unsqueeze(1) # (batch, 1)

        optimizer.zero_grad()
        logits = model(signals)                 # (batch, 1)
        loss   = criterion(logits, labels)
        loss.backward()

        # Gradient clipping for stability (mild)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Evaluate on a DataLoader.

    Returns:
        mean_loss, auroc, all_probs, all_labels
    """
    model.eval()
    total_loss = 0.0
    n_batches  = 0
    all_probs  = []
    all_labels = []

    for signals, labels in loader:
        signals = signals.to(device)
        labels_d = labels.to(device).unsqueeze(1)

        logits = model(signals)
        loss   = criterion(logits, labels_d)

        probs  = torch.sigmoid(logits).squeeze(1)   # (batch,)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.numpy())

        total_loss += loss.item()
        n_batches  += 1

    all_probs  = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    mean_loss = total_loss / max(n_batches, 1)
    try:
        auroc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auroc = 0.5   # edge case: only one class in tiny val fold

    return mean_loss, auroc, all_probs, all_labels


# =============================================================================
# EMBEDDING EXTRACTION
# =============================================================================

@torch.no_grad()
def extract_embeddings(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract penultimate-layer embeddings and probabilities.

    Returns:
        embeddings: (N, embed_dim)
        probs:      (N,)
        labels:     (N,)
    """
    model.eval()
    all_embeddings = []
    all_probs      = []
    all_labels     = []

    for signals, labels in loader:
        signals = signals.to(device)

        logits, embeds = model.forward_with_embedding(signals)
        probs = torch.sigmoid(logits).squeeze(1)

        all_embeddings.append(embeds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.numpy())

    return (
        np.concatenate(all_embeddings),   # (N, embed_dim)
        np.concatenate(all_probs),        # (N,)
        np.concatenate(all_labels),       # (N,)
    )


# =============================================================================
# FULL FOLD TRAINING
# =============================================================================

def train_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    fold_id: int,
    model_save_path: str,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-3,
    max_epochs: int = 100,
    patience: int = 15,
    pos_weight: float = 3.776,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, Dict]:
    """
    Train a model for one fold with early stopping.

    Args:
        model:           freshly initialized BrugadaCNN
        train_loader:    DataLoader for training fold
        val_loader:      DataLoader for validation fold (no augmentation)
        fold_id:         0-4
        model_save_path: where to save the best checkpoint (.pt)
        ...

    Returns:
        model:        best checkpoint loaded
        fold_history: dict with training curves and final best metrics
    """
    if device is None:
        device = torch.device("cpu")

    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32).to(device)
    )

    early_stop = EarlyStopping(patience=patience)
    history    = {"train_loss": [], "val_loss": [], "val_auroc": []}

    logger.info(f"\nFold {fold_id} | Training started | "
                f"lr={learning_rate} wd={weight_decay} "
                f"pos_weight={pos_weight.item():.4f} max_epochs={max_epochs}")

    t0 = time.time()

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_auroc, _, _ = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auroc"].append(val_auroc)

        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - t0
            logger.info(f"  Fold {fold_id} | Epoch {epoch:3d}/{max_epochs} | "
                        f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                        f"val_auroc={val_auroc:.4f} | {elapsed:.0f}s")

        should_stop = early_stop.step(val_auroc, model, epoch)
        if should_stop:
            logger.info(f"  Early stopping at epoch {epoch} "
                        f"(best epoch={early_stop.best_epoch}, "
                        f"best_auroc={early_stop.best_score:.4f})")
            break

    # Restore best weights
    model = early_stop.restore_best(model)

    # Save best checkpoint
    ensure_dirs(os.path.dirname(model_save_path))
    torch.save(model.state_dict(), model_save_path)
    logger.info(f"  Saved best model: {model_save_path}")

    history["best_epoch"]    = early_stop.best_epoch
    history["best_val_auroc"] = early_stop.best_score
    history["total_epochs"]  = epoch

    return model, history