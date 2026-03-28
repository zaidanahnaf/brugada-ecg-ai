# run_handoff.py

import logging
from catboost import CatBoostClassifier
import pandas as pd
from pathlib import Path

from src.config import CFG
from src.data.data_loader import load_metadata
from src.fold_manager import load_folds
from src.feature_store import (
    generate_full_manifest,
    export_scaled_feature_matrices,
    export_top_k_feature_set,
    save_oof_predictions
)
from src.handoff_validation import validate_all_handoff_files
from experiments.hybrid_fusion import (
    run_early_fusion_cv,
    run_late_fusion_cv,
    run_stacking_fusion_cv
)
from experiments.hybrid_fusion import save_integration_contract
from sklearn.linear_model import LogisticRegression

logging.basicConfig(level=logging.INFO)

# ── Load ──────────────────────────────────────────────────────────
feature_df  = pd.read_csv("features/feature_matrix.csv")
fold_df     = load_folds("data/splits/fold_assignments.csv")
metadata    = load_metadata(CFG.metadata_path)


random_seed = 42
feature_cols = [
    c for c in feature_df.columns
    if c not in ['patient_id', CFG.target_col,
                 'pipeline_status', 'n_valid_beats']
]

def best_model_factory():
    return CatBoostClassifier(
        learning_rate=0.1,
        l2_leaf_reg=3,
        iterations=100,
        depth=3,
        border_count=32,
        auto_class_weights='Balanced',
        eval_metric='AUC',
        random_state=CFG.random_seed,
        verbose=0
    )

# ── Step 1: Generate Feature Manifest ────────────────────────────
print("[1/6] Generating feature manifest...")
manifest = generate_full_manifest(
    feature_df=feature_df,
    fold_dependent_features=[],
    save_path="features/feature_manifest.json"
)

# ── Step 2: Export Scaled Matrices ───────────────────────────────
print("[2/6] Exporting fold-safe scaled matrices...")
export_scaled_feature_matrices(
    feature_df=feature_df,
    fold_df=fold_df,
    feature_cols=feature_cols,
    output_dir="features/scaled"
)

# ── Step 3: Export Top-K Feature Sets ────────────────────────────
print("[3/6] Exporting reduced feature sets...")
try:
    consensus = pd.read_csv("results/interpretability/consensus_importance.csv")
    export_top_k_feature_set(
        feature_df=feature_df,
        consensus_importance=consensus,
        k_values=[10, 20, 35],
        output_dir="features/reduced"
    )
except FileNotFoundError:
    print("  Consensus importance not yet computed — skipping top-K export")

# ── Step 4: Save OOF Predictions ─────────────────────────────────
print("[4/6] Generating OOF predictions...")
save_oof_predictions(
    fold_df=fold_df,
    feature_df=feature_df,
    model_factory=best_model_factory,
    feature_cols=feature_cols,
    model_name="CatBoostClassifier_depth3_border32",
    save_path="features/oof_predictions_handcrafted.csv"
)

# ── Step 5: Save Integration Contract ────────────────────────────
print("[5/6] Saving integration contract...")
save_integration_contract("INTEGRATION_CONTRACT.md")

# ── Step 6: Run Validation ────────────────────────────────────────
print("[6/6] Running handoff validation...")
validation_results = validate_all_handoff_files()

all_passed = all(validation_results.values())
if all_passed:
    print("\n✓ Handoff complete. Person 2 may proceed with hybrid experiments.")
else:
    print("\n✗ Handoff incomplete. Resolve validation failures before fusion.")

# ── Optional: Run Fusion Experiments (after CNN files arrive) ────
cnn_emb_path = "features/cnn_embeddings.csv"
cnn_oof_path = "features/cnn_fold_probs.csv"

if Path(cnn_emb_path).exists() and Path(cnn_oof_path).exists():
    print("\nCNN files detected. Running hybrid fusion experiments...")

    cnn_df = pd.read_csv(cnn_emb_path)
    cnn_oof = pd.read_csv(cnn_oof_path)
    hc_oof  = pd.read_csv("features/oof_predictions_handcrafted.csv")

    cnn_cols = [c for c in cnn_df.columns if c.startswith('cnn_embed_')]
    top20    = pd.read_json(
        "features/reduced/top_k_20_feature_names.json"
    )['feature_names'].tolist()

    early_result  = run_early_fusion_cv(
        feature_df, cnn_df, fold_df, top20, cnn_cols,
        meta_model_factory=best_model_factory,
        results_dir="results/hybrid/early_fusion"
    )
    late_result   = run_late_fusion_cv(
        hc_oof, cnn_oof, fold_df,
        weight_method='auroc_weighted',
        results_dir="results/hybrid/late_fusion"
    )
    stack_result  = run_stacking_fusion_cv(
        hc_oof, cnn_oof, fold_df,
        results_dir="results/hybrid/stacking"
    )

    print("\n── HYBRID FUSION RESULTS ──")
    for name, res in [
        ('Early Fusion',  early_result),
        ('Late Fusion',   late_result),
        ('Stacking',      stack_result)
    ]:
        print(
            f"  {name:<18}: "
            f"AUROC={res.get('default_auroc_mean', 0):.3f} ± "
            f"{res.get('default_auroc_std', 0):.3f}"
        )
else:
    print(f"\nCNN files not yet available at {cnn_emb_path}")
    print("Handoff files are ready. Waiting for Person 2.")