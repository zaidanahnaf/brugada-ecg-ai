"""
train.py — 5-fold cross-validation training loop.

Key design decisions:
  1. Uses fold_assignments.csv as SINGLE SOURCE OF TRUTH (SHA256 verified)
  2. Trains on beat-level, aggregates predictions to subject-level for eval
  3. Normalization fit on TRAINING fold only -> no leakage
  4. Weighted BCEWithLogitsLoss for class imbalance (pos_weight ≈ 3.77)
  5. Early stopping on validation AUROC
  6. Saves OOF embeddings and probabilities for Person 4 fusion
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.beat_segmentation import segment_all_recordings
from src.config import (
    BATCH_SIZE,
    BEAT_AGGREGATION,
    EMBEDDING_DIM,
    EPOCHS,
    FEATURES_DIR,
    FOLD_ASSIGNMENTS,
    LEARNING_RATE,
    LOGS_DIR,
    MODELS_DIR,
    N_FOLDS,
    N_NEGATIVE,
    N_POSITIVE,
    PATIENCE,
    POS_WEIGHT,
    PROBLEMATIC_SUBJECTS,
    RESULTS_DIR,
    SEED,
    WEIGHT_DECAY,
    MODEL_VARIANTS,
    set_all_seeds,
)
from src.dataset import BeatDataset, RecordingDataset, collate_with_pid
from src.evaluate import (
    aggregate_beat_predictions,
    compute_metrics,
    find_youden_threshold,
)
from src.load_data import load_all_signals, load_fold_assignments, load_metadata
from src.model import build_model
from src.preprocess import build_preprocessed_dataset

logger = logging.getLogger(__name__)


# ============================================================
# EARLY STOPPING
# ============================================================

class EarlyStopping:
    def __init__(self, patience: int = PATIENCE, mode: str = "max"):
        self.patience = patience
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.best_state = None
        self.stopped = False

    def step(self, score: float, model: nn.Module) -> bool:
        """Returns True if training should stop."""
        improved = (
            self.best_score is None
            or (self.mode == "max" and score > self.best_score)
            or (self.mode == "min" and score < self.best_score)
        )
        if improved:
            self.best_score = score
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stopped = True
        return self.stopped

    def restore_best(self, model: nn.Module):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# ============================================================
# ONE TRAINING EPOCH
# ============================================================

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
    n_batches = 0

    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, _ = model(x)
        loss = criterion(logits.squeeze(1), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


# ============================================================
# VALIDATION — BEAT-LEVEL INFERENCE -> SUBJECT-LEVEL AGGREGATION
# ============================================================

@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    labels_dict: Dict[str, int],
    aggregation: str = BEAT_AGGREGATION,
) -> Tuple[float, float, Dict[str, float], Dict[str, np.ndarray]]:
    """
    Run validation:
      1. Collect beat-level logits + patient_ids
      2. Aggregate to subject-level probabilities
      3. Compute AUROC

    Returns:
        val_loss, auroc, subject_probs_dict, subject_embeddings_dict
    """
    model.eval()
    all_logits:     List[float] = []
    all_pids:       List[str]   = []
    all_embeddings: List[np.ndarray] = []
    total_loss = 0.0
    n_batches = 0

    for x, y, pids in loader:
        x, y = x.to(device), y.to(device)
        logits, embeddings = model(x, return_embedding=True)
        loss = criterion(logits.squeeze(1), y)
        total_loss += loss.item()
        n_batches += 1

        all_logits.extend(logits.squeeze(1).cpu().numpy().tolist())
        all_pids.extend(pids)
        if embeddings is not None:
            all_embeddings.extend(embeddings.cpu().numpy())

    val_loss = total_loss / max(n_batches, 1)

    # Aggregate beat probabilities -> subject-level
    beat_probs = torch.sigmoid(torch.tensor(all_logits)).numpy()
    subject_probs, subject_embeddings = aggregate_beat_predictions(
        beat_probs, all_embeddings, all_pids, aggregation=aggregation
    )

    # Compute AUROC
    val_ids    = list(subject_probs.keys())
    y_true     = np.array([labels_dict[pid] for pid in val_ids])
    y_prob     = np.array([subject_probs[pid] for pid in val_ids])
    metrics    = compute_metrics(y_true, y_prob)
    auroc      = metrics.get("auroc", 0.0)

    return val_loss, auroc, subject_probs, subject_embeddings


# ============================================================
# TRAIN ONE FOLD
# ============================================================

def train_fold(
    fold_id:       int,
    train_ids:     List[str],
    val_ids:       List[str],
    labels_dict:   Dict[str, int],
    all_signals:   Dict[str, Optional[np.ndarray]],
    lead_indices:  List[int],
    variant_name:  str,
    device:        torch.device,
    model_type:    str = "cnn",
    lr:            float = LEARNING_RATE,
    batch_size:    int   = BATCH_SIZE,
    epochs:        int   = EPOCHS,
    models_dir:    Path  = MODELS_DIR,
) -> Tuple[Dict, Dict[str, float], Dict[str, np.ndarray]]:
    """
    Train model for a single fold.

    Returns:
        metrics_dict, oof_probs_dict {pid->prob}, oof_embeddings_dict {pid->emb}
    """
    set_all_seeds(SEED + fold_id)  # vary seed per fold for diversity

    # ---------- 1. Preprocess (fit on training fold only) ----------
    train_sigs_raw = {pid: all_signals.get(pid) for pid in train_ids}
    val_sigs_raw   = {pid: all_signals.get(pid) for pid in val_ids}

    norm_path = LOGS_DIR / f"normalizer_fold{fold_id}_{variant_name}.npz"
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    train_preprocessed, val_preprocessed, _ = build_preprocessed_dataset(
        train_sigs_raw, val_sigs_raw,
        fold_id=fold_id,
        save_normalizer_path=str(norm_path),
    )

    # ---------- 2. Beat segmentation ----------
    train_beats = segment_all_recordings(train_preprocessed, lead_indices=lead_indices)
    val_beats   = segment_all_recordings(val_preprocessed,   lead_indices=lead_indices)

    # ---------- 3. Build datasets ----------
    train_dataset = BeatDataset(train_beats, labels_dict, train_ids, augment=True)
    val_dataset   = BeatDataset(val_beats,   labels_dict, val_ids,   augment=False)

    if len(train_dataset) == 0:
        raise RuntimeError(f"[Fold {fold_id}] Training dataset is empty!")
    if len(val_dataset) == 0:
        raise RuntimeError(f"[Fold {fold_id}] Validation dataset is empty!")

    # ---------- 4. DataLoaders ----------
    # Weighted sampler for class balance
    beat_labels = train_dataset.get_labels()
    n_pos = sum(beat_labels)
    n_neg = len(beat_labels) - n_pos
    weights = [N_NEGATIVE / n_neg if l == 0 else N_POSITIVE / n_pos for l in beat_labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler,
        collate_fn=collate_with_pid, num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_with_pid, num_workers=2, pin_memory=True,
    )

    # ---------- 5. Model, optimizer, criterion ----------
    n_leads = len(lead_indices)
    model = build_model(model_type=model_type, n_leads=n_leads).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    pos_weight = torch.tensor([POS_WEIGHT], dtype=torch.float32).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    early_stop = EarlyStopping(patience=PATIENCE, mode="max")

    # ---------- 6. Training loop ----------
    history = {"train_loss": [], "val_loss": [], "val_auroc": []}
    best_auroc = 0.0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, auroc, subj_probs, subj_embs = validate(
            model, val_loader, criterion, device, labels_dict
        )
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auroc"].append(auroc)

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                f"[Fold {fold_id}/{variant_name}] Epoch {epoch:3d} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"val_auroc={auroc:.4f}"
            )

        if early_stop.step(auroc, model):
            logger.info(
                f"[Fold {fold_id}/{variant_name}] Early stopping at epoch {epoch}. "
                f"Best val AUROC={early_stop.best_score:.4f}"
            )
            break

    # Restore best weights
    early_stop.restore_best(model)
    best_auroc = early_stop.best_score or auroc

    # ---------- 7. Save model ----------
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"cnn_{variant_name}_fold_{fold_id}.pt"
    torch.save(model.state_dict(), model_path)
    logger.info(f"[Fold {fold_id}/{variant_name}] Model saved -> {model_path}")

    # ---------- 8. Final OOF inference (best model) ----------
    _, _, oof_probs, oof_embeddings = validate(
        model, val_loader, criterion, device, labels_dict
    )

    # ---------- 9. Compute final metrics on val fold ----------
    val_ids_list = list(oof_probs.keys())
    y_true = np.array([labels_dict[pid] for pid in val_ids_list])
    y_prob = np.array([oof_probs[pid]   for pid in val_ids_list])
    metrics = compute_metrics(y_true, y_prob)
    metrics["fold_id"]    = fold_id
    metrics["variant"]    = variant_name
    metrics["n_train"]    = len(train_dataset)
    metrics["n_val_subj"] = len(val_ids_list)
    metrics["epochs_run"] = len(history["train_loss"])

    logger.info(
        f"[Fold {fold_id}/{variant_name}] "
        f"AUROC={metrics['auroc']:.4f} | "
        f"Sensitivity={metrics.get('sensitivity', 0):.4f} | "
        f"Specificity={metrics.get('specificity', 0):.4f}"
    )

    return metrics, oof_probs, oof_embeddings


# ============================================================
# FULL 5-FOLD CROSS-VALIDATION
# ============================================================

def run_cross_validation(
    variant_name: str = "12lead",
    model_type:   str = "cnn",
    lr:           float = LEARNING_RATE,
    batch_size:   int   = BATCH_SIZE,
    epochs:       int   = EPOCHS,
    device:       Optional[torch.device] = None,
) -> Dict:
    """
    Run full 5-fold cross-validation for one model variant.

    Args:
        variant_name: "3lead" or "12lead"
        model_type:   "cnn" or "resnet"
        lr, batch_size, epochs: training hyperparameters
        device: torch device (auto-detected if None)

    Returns:
        Summary dict with mean/std metrics and all OOF predictions.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    set_all_seeds(SEED)

    # ---------- Setup dirs ----------
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- Load fold assignments (verifies SHA256) ----------
    folds_df = load_fold_assignments(FOLD_ASSIGNMENTS)
    labels_dict = dict(zip(folds_df["patient_id"].astype(str),
                       folds_df["brugada"].astype(int)))

    # ---------- Load all signals once ----------
    all_pids = folds_df["patient_id"].astype(str).tolist()
    logger.info(f"Loading {len(all_pids)} signals...")
    all_signals = load_all_signals(all_pids)

    # ---------- Lead indices ----------
    lead_indices = MODEL_VARIANTS[variant_name]["lead_indices"]
    logger.info(f"Variant: {variant_name} | lead_indices: {lead_indices}")

    # ---------- 5-fold loop ----------
    all_fold_metrics   = []
    all_oof_probs      = {}   # {pid -> prob}
    all_oof_embeddings = {}   # {pid -> emb}
    all_oof_fold_ids   = {}   # {pid -> fold_id}

    for fold_id in range(N_FOLDS):
        fold_df    = folds_df[folds_df["fold_id"] == fold_id]
        val_ids    = fold_df["patient_id"].astype(str).tolist()
        train_ids  = folds_df[folds_df["fold_id"] != fold_id]["patient_id"].astype(str).tolist()

        logger.info(
            f"\n{'='*60}\n"
            f"FOLD {fold_id} | train={len(train_ids)}, val={len(val_ids)}\n"
            f"{'='*60}"
        )

        metrics, oof_probs, oof_embeddings = train_fold(
            fold_id=fold_id,
            train_ids=train_ids,
            val_ids=val_ids,
            labels_dict=labels_dict,
            all_signals=all_signals,
            lead_indices=lead_indices,
            variant_name=variant_name,
            device=device,
            model_type=model_type,
            lr=lr,
            batch_size=batch_size,
            epochs=epochs,
        )

        all_fold_metrics.append(metrics)
        all_oof_probs.update(oof_probs)
        all_oof_embeddings.update(oof_embeddings)
        for pid in val_ids:
            all_oof_fold_ids[pid] = fold_id

    # ---------- Aggregate results ----------
    summary = _compute_cv_summary(
        all_fold_metrics,
        all_oof_probs,
        all_oof_embeddings,
        all_oof_fold_ids,
        labels_dict,
        all_pids,
        variant_name,
        model_type,
    )

    return summary


# ============================================================
# SUMMARY COMPUTATION + SAVING DELIVERABLES
# ============================================================

def _compute_cv_summary(
    fold_metrics:      List[Dict],
    oof_probs:         Dict[str, float],
    oof_embeddings:    Dict[str, np.ndarray],
    oof_fold_ids:      Dict[str, int],
    labels_dict:       Dict[str, int],
    all_pids:          List[str],
    variant_name:      str,
    model_type:        str,
) -> Dict:
    """Aggregate fold metrics and save deliverable files."""
    from src.model import build_model

    keys = ["auroc", "auprc", "sensitivity", "specificity", "f1_positive", "brier_score"]
    summary = {}
    for k in keys:
        vals = [m[k] for m in fold_metrics if k in m]
        if vals:
            summary[f"{k}_mean"] = float(np.mean(vals))
            summary[f"{k}_std"]  = float(np.std(vals))

    # Get n_parameters from first fold model (architecture is same)
    n_leads = MODEL_VARIANTS[variant_name]["n_leads"]
    tmp_model = build_model(model_type=model_type, n_leads=n_leads)
    summary["n_parameters"] = tmp_model.count_parameters()
    del tmp_model

    summary["variant"]      = variant_name
    summary["model_type"]   = model_type
    summary["best_params"]  = {
        "lr": LEARNING_RATE, "batch_size": BATCH_SIZE,
        "epochs": EPOCHS, "patience": PATIENCE,
    }

    logger.info(
        f"\n{'='*60}\n"
        f"CV Summary [{variant_name}]\n"
        + "\n".join([
            f"  {k}: {summary.get(f'{k}_mean', 0):.4f} ± {summary.get(f'{k}_std', 0):.4f}"
            for k in keys
        ])
        + f"\n{'='*60}"
    )

    # ---------- Save cnn_fold_probs.csv ----------
    rows = []
    for pid in all_pids:
        pid_str = str(pid)
        rows.append({
            "patient_id":       pid_str,
            "brugada":          labels_dict.get(pid_str, np.nan),
            "oof_prob_brugada": oof_probs.get(pid_str, np.nan),
            "fold_id":          oof_fold_ids.get(pid_str, np.nan),
        })
    probs_df = pd.DataFrame(rows)
    probs_path = FEATURES_DIR / f"cnn_fold_probs_{variant_name}.csv"
    probs_df.to_csv(probs_path, index=False)
    logger.info(f"Saved OOF probs -> {probs_path}")

    # ---------- Save cnn_embeddings.csv ----------
    if oof_embeddings:
        emb_dim = len(next(iter(oof_embeddings.values())))
        emb_rows = []
        for pid in all_pids:
            pid_str = str(pid)
            emb = oof_embeddings.get(pid_str, None)
            row = {"patient_id": pid_str}
            if emb is not None:
                for i, v in enumerate(emb):
                    row[f"cnn_embed_{i}"] = float(v)
            else:
                for i in range(emb_dim):
                    row[f"cnn_embed_{i}"] = np.nan
            emb_rows.append(row)
        emb_df = pd.DataFrame(emb_rows)
        emb_path = FEATURES_DIR / f"cnn_embeddings_{variant_name}.csv"
        emb_df.to_csv(emb_path, index=False)
        logger.info(f"Saved OOF embeddings -> {emb_path}")

    # ---------- Save cnn_cv_summary.json ----------
    summary_path = RESULTS_DIR / f"cnn_cv_summary_{variant_name}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved CV summary -> {summary_path}")

    return summary


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import argparse
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Train Brugada CNN")
    parser.add_argument("--variant",    default="12lead", choices=["3lead", "12lead"])
    parser.add_argument("--model_type", default="cnn",    choices=["cnn", "resnet"])
    parser.add_argument("--lr",         type=float, default=LEARNING_RATE)
    parser.add_argument("--batch_size", type=int,   default=BATCH_SIZE)
    parser.add_argument("--epochs",     type=int,   default=EPOCHS)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    run_cross_validation(
        variant_name=args.variant,
        model_type=args.model_type,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        device=device,
    )