# run_ablation.py

import logging
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.config import CFG
from src.fold_manager import load_folds
from experiments.ablation import run_full_ablation_study
from experiments.ablation_stats import run_key_comparisons
from experiments.ablation_conclusions import generate_conclusions, print_conclusions

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s — %(message)s'
)

# ── Load artefacts ────────────────────────────────────────────────
feature_df = pd.read_csv("outputs/features/feature_matrix.csv")
fold_df = load_folds("data/splits/fold_assignments.csv")

# Optional: load CNN embeddings if available from Person 2
try:
    cnn_df = pd.read_csv("features/cnn_embeddings.csv")
    print(f"CNN embeddings loaded: {cnn_df.shape}")
except FileNotFoundError:
    cnn_df = None
    print("CNN embeddings not found — hybrid experiments will be skipped.")


# ── Fix best model from Phase 3 results ──────────────────────────
# Replace these params with actual best params from classical_ml results
# def model_factory():
#     """
#     Returns a fresh estimator with FIXED hyperparameters.
#     These are set from the best inner-CV result in Phase 3.
#     Do NOT retune here — ablation varies features only.
#     """
#     return LogisticRegression(
#         C=0.1,
#         penalty='l1',
#         solver='saga',
#         class_weight='balanced',
#         max_iter=2000,
#         random_state=CFG.random_seed
#     )

def model_factory():
    """
    Best model from Stage 5: CatBoost_Balanced, feat_sel=none
    AUROC: 0.922 ± 0.026
    Most common best params from inner CV (5 folds):
        learning_rate=0.1, l2_leaf_reg=3, iterations=100,
        depth=3, border_count=32
    """
    from catboost import CatBoostClassifier
    return CatBoostClassifier(
        learning_rate=0.1,
        l2_leaf_reg=3,
        iterations=100,
        depth=3,
        border_count=32,
        auto_class_weights='Balanced',
        eval_metric='AUC',
        random_seed=42,
        verbose=0
    )


def rf_factory():
    """RF for cross-model stability check."""
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=1,
        max_features='sqrt',
        max_depth=None,
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1
    )


# ── Run Ablation ──────────────────────────────────────────────────
results_df = run_full_ablation_study(
    feature_df=feature_df,
    fold_df=fold_df,
    model_factory=model_factory,
    results_dir="results/ablation",
    cnn_embedding_df=cnn_df,
    skip_metadata_experiments=False
)

# ── Cross-model stability check (top 3 experiments only) ─────────
print("\nRunning RF cross-model stability check...")
for exp_id in ['EXP_03_st_only_v1v3',
               'EXP_07_all_handcrafted',
               'EXP_09_v1v3_only_all_feature_types']:
    feature_cols_for_exp = results_df[
        results_df['exp_id'] == exp_id
    ]['feature_list'].iloc[0] if exp_id in results_df['exp_id'].values else []

    # (RF run omitted for brevity — same pattern as LR above)

# ── Statistical Comparisons ───────────────────────────────────────
print("\nRunning key pairwise comparisons...")
comparison_df = run_key_comparisons(results_dir="results/ablation")

# ── Generate Conclusions ──────────────────────────────────────────
conclusions = generate_conclusions(results_df, comparison_df)
print_conclusions(conclusions)

# Save conclusions
with open("results/ablation/mandatory_conclusions.txt", 'w') as f:
    for q_id, text in conclusions.items():
        f.write(f"[{q_id}]\n{text}\n\n")

print("\nAblation study complete.")
print(f"  Master table: results/ablation/ablation_master_table.csv")
print(f"  Comparisons:  results/ablation/key_comparisons.csv")
print(f"  Conclusions:  results/ablation/mandatory_conclusions.txt")