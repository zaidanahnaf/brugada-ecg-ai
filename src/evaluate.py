"""
evaluate.py — Metrics computation, beat aggregation, and threshold selection.

All evaluation happens at SUBJECT-LEVEL (1 probability per patient).
Beat-level outputs from the CNN must be aggregated here before metrics.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    auc,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.config import BEAT_AGGREGATION

logger = logging.getLogger(__name__)


# ============================================================
# BEAT -> SUBJECT AGGREGATION
# ============================================================

def aggregate_beat_predictions(
    beat_probs:      np.ndarray,         # (n_beats,)
    beat_embeddings: List[np.ndarray],   # list of (emb_dim,) arrays
    patient_ids:     List[str],          # (n_beats,) — one pid per beat
    aggregation:     str = BEAT_AGGREGATION,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """
    Aggregate beat-level probabilities and embeddings to subject-level.

    Aggregation strategies:
        "mean"   — simple mean of beat probabilities (recommended)
        "max"    — max of beat probabilities (more sensitive, more FP)
        "median" — median of beat probabilities

    Embeddings are always mean-aggregated (mean pooling over beats).

    Args:
        beat_probs:      sigmoid probabilities per beat
        beat_embeddings: embedding vectors per beat
        patient_ids:     which subject each beat belongs to
        aggregation:     how to combine beat probs

    Returns:
        subject_probs:      {patient_id -> float}
        subject_embeddings: {patient_id -> np.ndarray (emb_dim,)}
    """
    # Group beats by subject
    pid_to_probs: Dict[str, List[float]] = {}
    pid_to_embs:  Dict[str, List[np.ndarray]] = {}

    for i, pid in enumerate(patient_ids):
        if pid not in pid_to_probs:
            pid_to_probs[pid] = []
            pid_to_embs[pid]  = []
        pid_to_probs[pid].append(float(beat_probs[i]))
        if beat_embeddings and i < len(beat_embeddings):
            pid_to_embs[pid].append(beat_embeddings[i])

    subject_probs      = {}
    subject_embeddings = {}

    for pid, probs in pid_to_probs.items():
        arr = np.array(probs)
        if aggregation == "mean":
            subject_probs[pid] = float(arr.mean())
        elif aggregation == "max":
            subject_probs[pid] = float(arr.max())
        elif aggregation == "median":
            subject_probs[pid] = float(np.median(arr))
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

        embs = pid_to_embs.get(pid, [])
        if embs:
            subject_embeddings[pid] = np.mean(np.stack(embs, axis=0), axis=0)

    return subject_probs, subject_embeddings


# ============================================================
# YOUDEN THRESHOLD SELECTION
# ============================================================

def find_youden_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Find the optimal classification threshold using the Youden index.
    Youden J = Sensitivity + Specificity - 1

    Returns:
        (threshold, sensitivity_at_threshold, specificity_at_threshold)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    specificity = 1.0 - fpr
    youden_j = tpr + specificity - 1.0
    best_idx = np.argmax(youden_j)

    best_threshold   = float(thresholds[best_idx])
    best_sensitivity = float(tpr[best_idx])
    best_specificity = float(specificity[best_idx])

    return best_threshold, best_sensitivity, best_specificity


# ============================================================
# FULL METRICS SUITE
# ============================================================

def compute_metrics(
    y_true:    np.ndarray,
    y_prob:    np.ndarray,
    threshold: Optional[float] = None,
) -> Dict[str, float]:
    """
    Compute all required metrics for a fold.

    Args:
        y_true:    binary labels (0/1)
        y_prob:    predicted probabilities
        threshold: if None, Youden index is used

    Returns:
        dict with keys matching cnn_cv_summary.json schema.
    """
    # --- Sanitise inputs ---
    y_true = np.array(y_true, dtype=float).ravel()
    y_prob = np.array(y_prob, dtype=float).ravel()

    if len(y_true) != len(y_prob):
        raise ValueError(
            f"y_true length ({len(y_true)}) != y_prob length ({len(y_prob)})"
        )

    # Drop any rows where either y_true or y_prob is NaN/Inf
    valid_mask = np.isfinite(y_true) & np.isfinite(y_prob)
    n_dropped = np.sum(~valid_mask)
    if n_dropped > 0:
        logger.warning(
            f"compute_metrics: dropping {n_dropped} subject(s) with NaN/Inf "
            f"values (likely failed beat segmentation)."
        )
    y_true = y_true[valid_mask].astype(int)
    y_prob = y_prob[valid_mask]

    if len(y_true) == 0:
        logger.error("compute_metrics: no valid samples after NaN filtering.")
        return {"auroc": 0.5, "auprc": 0.0, "sensitivity": 0.0,
                "specificity": 0.0, "f1_positive": 0.0, "brier_score": 1.0,
                "youden_threshold": 0.5, "sens_at_90spec": 0.0}

    if len(np.unique(y_true)) < 2:
        logger.warning(
            "Only one class present in y_true — AUROC undefined. "
            "Returning placeholder values. This may happen in early smoke-test epochs."
        )
        return {"auroc": 0.5, "auprc": 0.0, "sensitivity": 0.0,
                "specificity": 0.0, "f1_positive": 0.0, "brier_score": 1.0,
                "youden_threshold": 0.5, "sens_at_90spec": 0.0}

    # --- Threshold-free metrics ---
    auroc = float(roc_auc_score(y_true, y_prob, multi_class="ovr"))
    auprc = float(average_precision_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))

    # --- Threshold-based metrics ---
    if threshold is None:
        threshold, sensitivity, specificity = find_youden_threshold(y_true, y_prob)
    else:
        y_pred = (y_prob >= threshold).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        sensitivity = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)

    y_pred = (y_prob >= threshold).astype(int)
    f1_pos = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))

    # --- Sensitivity at 90% specificity ---
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_prob)
    spec_arr = 1.0 - fpr
    # Find threshold closest to 90% specificity
    idx_90spec = np.argmin(np.abs(spec_arr - 0.90))
    sens_at_90spec = float(tpr[idx_90spec])

    return {
        "auroc":            auroc,
        "auprc":            auprc,
        "sensitivity":      float(sensitivity),
        "specificity":      float(specificity),
        "f1_positive":      f1_pos,
        "brier_score":      brier,
        "youden_threshold": float(threshold),
        "sens_at_90spec":   sens_at_90spec,
    }


def print_fold_metrics(fold_id: int, metrics: Dict[str, float], variant: str = "") -> None:
    """Pretty-print metrics for one fold."""
    tag = f"[Fold {fold_id}]" + (f"[{variant}]" if variant else "")
    print(
        f"{tag} "
        f"AUROC={metrics.get('auroc', 0):.4f} | "
        f"AUPRC={metrics.get('auprc', 0):.4f} | "
        f"Sens={metrics.get('sensitivity', 0):.4f} | "
        f"Spec={metrics.get('specificity', 0):.4f} | "
        f"F1={metrics.get('f1_positive', 0):.4f} | "
        f"Brier={metrics.get('brier_score', 0):.4f} | "
        f"Threshold={metrics.get('youden_threshold', 0):.4f}"
    )


def print_cv_summary(fold_metrics: List[Dict[str, float]], variant: str = "") -> None:
    """Print mean ± std across folds."""
    keys = ["auroc", "auprc", "sensitivity", "specificity", "f1_positive", "brier_score"]
    print(f"\n{'='*60}")
    print(f"CV Summary {'[' + variant + ']' if variant else ''}")
    print(f"{'='*60}")
    for k in keys:
        vals = [m[k] for m in fold_metrics if k in m]
        if vals:
            print(f"  {k:20s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    # Minimum target check
    auroc_mean = np.mean([m["auroc"] for m in fold_metrics])
    if auroc_mean < 0.85:
        print(f"\n  ⚠ AUROC {auroc_mean:.4f} < 0.85 target — revisit architecture/preprocessing")
    elif auroc_mean > 0.90:
        print(f"\n  ✓ AUROC {auroc_mean:.4f} > 0.90 — hybrid fusion highly promising")
    else:
        print(f"\n  ✓ AUROC {auroc_mean:.4f} within acceptable range")
    print(f"{'='*60}\n")


# ============================================================
# SANITY CHECK
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    np.random.seed(42)

    # Simulate beat-level predictions for 5 subjects, 10 beats each
    n_subj, n_beats = 5, 10
    emb_dim = 64
    labels = {f"sub_{i}": (1 if i < 2 else 0) for i in range(n_subj)}

    pids   = [pid for pid in labels for _ in range(n_beats)]
    probs  = np.clip(np.random.randn(len(pids)) * 0.2 + [0.7 if labels[p] else 0.3 for p in pids], 0, 1)
    embs   = [np.random.randn(emb_dim).astype(np.float32) for _ in pids]

    subj_probs, subj_embs = aggregate_beat_predictions(probs, embs, pids, aggregation="mean")
    print("Subject probs:", {k: f"{v:.3f}" for k, v in subj_probs.items()})

    y_true = np.array([labels[pid] for pid in subj_probs.keys()])
    y_prob = np.array(list(subj_probs.values()))
    metrics = compute_metrics(y_true, y_prob)
    print("Metrics:", {k: round(v, 4) for k, v in metrics.items()})