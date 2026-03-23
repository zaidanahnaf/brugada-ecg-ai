# scripts/get_catboost_best_params.py
"""
Jalankan CatBoost dengan feat_sel=none dan simpan best params.
Hanya 1 model, ~5-10 menit.
"""

import pandas as pd
import logging
from src.config import CFG
from src.fold_manager import load_folds
from src.feature_store import get_clean_feature_matrix
from experiments.classical_ml import run_cv_for_model
from experiments.model_registry import CATBOOST

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s — %(message)s')

df      = pd.read_csv("features/feature_matrix.csv")
fold_df = load_folds("data/splits/fold_assignments.csv")

feature_cols = [
    c for c in df.columns
    if c not in ['patient_id', CFG.target_col,
                 'pipeline_status', 'n_valid_beats']
    and not c.startswith('qc_')
]

df_clean, feat_clean = get_clean_feature_matrix(
    df, feature_cols, max_missing_pct=50.0
)

# Jalankan CatBoost dengan feat_sel=none — sama persis dengan yang menang
summary = run_cv_for_model(
    model_config=CATBOOST,
    feature_df=df_clean,
    fold_df=fold_df,
    feature_cols=feat_clean,
    imbalance_strategy='class_weight',
    feature_selection_method='none',    # ← ini yang menghasilkan 0.922
    k_features=None,
    threshold_method='youden',
    calibration_method='sigmoid',
    results_dir="results/primary"
)

print(f"\n{'='*50}")
print(f"CatBoost (feat_sel=none)")
print(f"  AUROC      : {summary.get('default_auroc_mean', 0):.3f} ± {summary.get('default_auroc_std', 0):.3f}")
print(f"  Best params: {summary.get('best_params_most_common', 'NOT SAVED')}")
print(f"{'='*50}")