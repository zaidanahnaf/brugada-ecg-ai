# tests/run_interpretability.py

# run_interpretability.py

import logging
import pandas as pd
from catboost import CatBoostClassifier

from src.config.__init__ import CFG
from src.fold_manager import load_folds
from src.data.data_loader import load_metadata
from experiments.interpretability.permutation_importance import (
    compute_permutation_importance_cv,
    compute_model_native_importance,
    build_consensus_importance
)
from experiments.interpretability.shap_analysis import (
    compute_shap_oof, compute_shap_summaries
)
from experiments.interpretability.shap_plots import (
    plot_shap_beeswarm, plot_shap_bar,
    plot_lead_importance, plot_shap_waterfall_cases
)
from experiments.interpretability.lead_analysis import (
    build_lead_importance_report, analyze_v1v2v3_contributions
)
from experiments.interpretability.clinical_report import (
    generate_clinical_interpretation_report
)
from experiments.interpretability.error_analysis import run_error_analysis

logging.basicConfig(level=logging.INFO)
RESULTS_DIR = "outputs/results/interpretability"

# ── Load data ─────────────────────────────────────────────────────
feature_df = pd.read_csv("outputs/features/feature_matrix.csv")
fold_df = load_folds("data/splits/fold_assignments.csv")
metadata = load_metadata(CFG.data.metadata_path)

feature_cols = [
    c for c in feature_df.columns
    if c not in ['patient_id', CFG.data.target_col,
                 'pipeline_status', 'n_valid_beats']
    and not c.startswith('basal_')
    and not c.startswith('sudden_')
]

# ── Fixed model (from Phase 3 best result) ────────────────────────
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


# ── Layer 1: Permutation Importance ──────────────────────────────
print("\n[1/6] Computing permutation importance...")
perm_summary = compute_permutation_importance_cv(
    model_factory=model_factory,
    feature_df=feature_df,
    fold_df=fold_df,
    feature_cols=feature_cols,
    n_repeats=30,
    results_dir=RESULTS_DIR
)

print("\n[1b] Computing RF native importance...")
native_importance = compute_model_native_importance(
    model_factory=model_factory,
    feature_df=feature_df,
    fold_df=fold_df,
    feature_cols=feature_cols,
    model_type='tree',
    results_dir=RESULTS_DIR
)

print("\n[1c] Building consensus importance ranking...")
consensus = build_consensus_importance(perm_summary, native_importance, top_k=25)
consensus.to_csv(f"{RESULTS_DIR}/consensus_importance.csv", index=False)


# ── Layer 2: SHAP Analysis ─────────────────────────────────────────
print("\n[2/6] Computing SHAP values (OOF)...")
shap_result = compute_shap_oof(
    model_factory=model_factory,
    feature_df=feature_df,
    fold_df=fold_df,
    feature_cols=feature_cols,
    model_type='tree',
    results_dir=RESULTS_DIR
)

print("[2b] Computing SHAP summaries...")
shap_summaries = compute_shap_summaries(shap_result, RESULTS_DIR)


# ── Layer 3: SHAP Visualizations ─────────────────────────────────
print("\n[3/6] Generating SHAP plots...")
if shap_result:
    plot_shap_beeswarm(
        shap_result, top_k=25,
        save_path=f"{RESULTS_DIR}/shap_beeswarm.png"
    )
    plot_shap_bar(
        shap_summaries['global'], top_k=20,
        save_path=f"{RESULTS_DIR}/shap_bar.png"
    )
    plot_lead_importance(
        shap_summaries['lead'],
        save_path=f"{RESULTS_DIR}/lead_importance.png"
    )
    for case_type in ['fn', 'fp', 'tp']:
        plot_shap_waterfall_cases(
            shap_result=shap_result,
            feature_df=feature_df,
            case_type=case_type,
            n_cases=3,
            model_factory=model_factory,
            fold_df=fold_df,
            feature_cols=feature_cols,
            save_dir=f"{RESULTS_DIR}/waterfall"
        )


# ── Layer 4: Lead-Level Analysis ──────────────────────────────────
print("\n[4/6] Lead-level importance analysis...")
lead_importance = build_lead_importance_report(
    perm_summary, shap_summaries, feature_cols
)
lead_importance.to_csv(f"{RESULTS_DIR}/lead_importance_table.csv", index=False)

v1v2v3_analysis = analyze_v1v2v3_contributions(shap_result, feature_cols)
v1v2v3_analysis.to_csv(f"{RESULTS_DIR}/v1v2v3_feature_type_analysis.csv", index=False)


# ── Layer 5: Clinical Report ───────────────────────────────────────
print("\n[5/6] Generating clinical interpretation report...")
report = generate_clinical_interpretation_report(
    global_shap=shap_summaries.get('global', pd.DataFrame()),
    perm_importance=perm_summary,
    lead_importance=lead_importance,
    consensus_importance=consensus,
    top_k=15,
    save_path=f"{RESULTS_DIR}/clinical_report.txt"
)
print(report[:2000])   # Preview first 2000 chars


# ── Layer 6: Error Analysis ────────────────────────────────────────
print("\n[6/6] Running error analysis...")
error_results = run_error_analysis(
    feature_df=feature_df,
    shap_result=shap_result,
    metadata=metadata,
    model_factory=model_factory,
    fold_df=fold_df,
    feature_cols=feature_cols,
    threshold=0.5,
    results_dir=RESULTS_DIR
)
print(error_results['narrative'])

print(f"\nInterpretability pipeline complete.")
print(f"All outputs saved to: {RESULTS_DIR}/")