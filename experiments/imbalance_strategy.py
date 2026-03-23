# experiments/imbalance_strategy.py
"""
THREE parallel imbalance strategies — compared in ablation.
All operate INSIDE the training fold only. Never before split.

STRATEGY A — Class Weights (default)
    - Pass class_weight='balanced' or scale_pos_weight to estimator
    - No data modification
    - Preserves all Brugada morphology diversity
    - Recommended default

STRATEGY B — Controlled Random Undersampling (training fold only)
    - Undersample majority class to 2:1 or 3:1 ratio
    - Never 1:1 (throws away too much Normal data)
    - Applied AFTER train/val split, BEFORE model fit
    - Risk: reduces effective N; may lose Normal morphology diversity
    - Use imblearn.under_sampling.RandomUnderSampler

STRATEGY C — Threshold Tuning on Validation Fold
    - Train without class weight modification
    - Tune decision threshold on validation fold
    - Youden index OR minimum sensitivity constraint
    - Post-hoc imbalance correction
    - Evaluate whether threshold tuning alone is sufficient
"""

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from typing import Tuple, Optional


def get_class_weights(y_train: np.ndarray) -> dict:
    """Compute balanced class weights from training labels."""
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    return dict(zip(classes.tolist(), weights.tolist()))


def undersample_training_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    strategy: float = 0.5,    # Ratio: n_minority / n_majority after sampling
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply random undersampling to training fold only.

    strategy=0.5 -> majority class sampled to 2× minority size
    strategy=1.0 -> balanced 1:1 (NOT recommended for this dataset)

    IMPORTANT:
    - Log original and post-sampling class distribution
    - Verify minority class count is unchanged
    - Never call this before outer fold split
    """
    try:
        from imblearn.under_sampling import RandomUnderSampler
    except ImportError:
        print("WARNING: imblearn not available. Skipping undersampling.")
        return X_train, y_train

    n_minority = int((y_train == 1).sum())
    n_majority_target = int(n_minority / strategy)

    rus = RandomUnderSampler(
        sampling_strategy={0: n_majority_target, 1: n_minority},
        random_state=random_seed
    )
    X_res, y_res = rus.fit_resample(X_train, y_train)

    print(
        f"  Undersampling: {len(y_train)} -> {len(y_res)} "
        f"({(y_res==1).sum()} pos, {(y_res==0).sum()} neg)"
    )
    return X_res, y_res