# src/fold_manager.py

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
from src.config import CFG


def create_folds(
    metadata: pd.DataFrame,
    n_folds: int = CFG.n_folds,
    random_seed: int = CFG.random_seed,
    output_dir: str = "data/splits"
) -> pd.DataFrame:
    """
    Create stratified, patient-level CV folds.
    Saves fold_assignments.csv for full reproducibility.

    Fold assignment is computed ONCE and saved.
    All subsequent experiments LOAD this file — never recompute.
    This is the single source of truth for all CV experiments.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    df = metadata[['patient_id', CFG.target_col]].copy()
    df = df.sort_values('patient_id').reset_index(drop=True)

    skf = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=random_seed
    )

    df['fold_id'] = -1
    y = df[CFG.target_col].values

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df, y)):
        df.loc[val_idx, 'fold_id'] = fold_idx

    # Sanity check: no subject in multiple folds
    assert df['fold_id'].min() == 0
    assert df['fold_id'].max() == n_folds - 1
    assert (df['fold_id'] == -1).sum() == 0

    # Log class distribution per fold
    for f in range(n_folds):
        fold_df = df[df['fold_id'] == f]
        pos = fold_df[CFG.target_col].sum()
        print(
            f"Fold {f}: {len(fold_df)} subjects, "
            f"{pos} positive ({100*pos/len(fold_df):.1f}%)"
        )

    save_path = f"{output_dir}/fold_assignments.csv"
    df.to_csv(save_path, index=False)
    print(f"Fold assignments saved: {save_path}")
    return df


def load_folds(path: str = "data/splits/fold_assignments.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def get_fold_split(
    fold_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    val_fold: int
) -> tuple:
    """
    Returns (X_train, y_train, X_val, y_val, train_ids, val_ids)
    for one fold.

    CRITICAL: Feature scaling must be fit ONLY on X_train.
    """
    train_ids = fold_df[fold_df['fold_id'] != val_fold]['patient_id'].values
    val_ids = fold_df[fold_df['fold_id'] == val_fold]['patient_id'].values

    feature_cols = [
        c for c in feature_df.columns
        if c not in ['patient_id', CFG.target_col, 'fold_id',
                     'pipeline_status', 'n_valid_beats']
        and not c.startswith(CFG.qc_feature_prefix)   # Exclude qc_ columns
    ]
    # qc_ columns remain in feature_df for error analysis access
    # but are never passed to ML models as input features

    train_df = feature_df[feature_df['patient_id'].isin(train_ids)]
    val_df = feature_df[feature_df['patient_id'].isin(val_ids)]

    X_train = train_df[feature_cols].values.astype(float)
    y_train = train_df[CFG.target_col].values.astype(int)
    X_val = val_df[feature_cols].values.astype(float)
    y_val = val_df[CFG.target_col].values.astype(int)

    return X_train, y_train, X_val, y_val, train_ids, val_ids, feature_cols