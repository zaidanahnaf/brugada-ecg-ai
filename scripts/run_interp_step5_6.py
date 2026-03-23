# scripts/run_interp_step5_6.py

import pandas as pd
import logging
from src.config import CFG
from src.fold_manager import load_folds
from src.data.data_loader import load_metadata
from catboost import CatBoostClassifier

logging.basicConfig(level=logging.INFO)
RESULTS_DIR = "results/interpretability"

feature_df = pd.read_csv("features/feature_matrix.csv")
fold_df    = load_folds("data/splits/fold_assignments.csv")
metadata   = load_metadata(CFG.metadata_path)

feature_cols = [
    c for c in feature_df.columns
    if c not in ['patient_id', CFG.target_col,
                 'pipeline_status', 'n_valid_beats']
    and not c.startswith('qc_')
]

def model_factory():
    return CatBoostClassifier(
        learning_rate=0.1, l2_leaf_reg=3,
        iterations=100, depth=3, border_count=32,
        auto_class_weights='Balanced',
        eval_metric='AUC', random_seed=42, verbose=0
    )

# Load hasil yang sudah ada
from experiments.interpretability.permutation_importance import build_consensus_importance
from experiments.interpretability.shap_analysis import compute_shap_summaries
from experiments.interpretability.lead_analysis import build_lead_importance_report
from experiments.interpretability.clinical_report import generate_clinical_interpretation_report
from experiments.interpretability.error_analysis import run_error_analysis

import pandas as pd
import numpy as np

perm_summary = pd.read_csv(f"{RESULTS_DIR}/permutation_importance.csv")
native_df    = pd.read_csv(f"{RESULTS_DIR}/native_importance_tree.csv")
consensus    = build_consensus_importance(perm_summary, native_df, top_k=25)
consensus.to_csv(f"{RESULTS_DIR}/consensus_importance.csv", index=False)

# Load SHAP
shap_df = pd.read_csv(f"{RESULTS_DIR}/shap_values_oof.csv")
y_oof   = shap_df['label'].values
ids_oof = shap_df['patient_id'].values
feat_cols_shap = [c for c in shap_df.columns
                  if c not in ['patient_id', 'label']]
shap_vals = shap_df[feat_cols_shap].values

shap_result = {
    'shap_values':   shap_vals,
    'X_oof':         shap_vals,   # approximation for plots
    'y_oof':         y_oof,
    'patient_ids':   ids_oof,
    'feature_names': feat_cols_shap
}

shap_summaries = compute_shap_summaries(shap_result, RESULTS_DIR)
lead_importance = build_lead_importance_report(
    perm_summary, shap_summaries, feat_cols_shap
)
lead_importance.to_csv(f"{RESULTS_DIR}/lead_importance_table.csv", index=False)

# Step 5 — Clinical report (with utf-8 fix)
print("\n[5/6] Generating clinical interpretation report...")
report = generate_clinical_interpretation_report(
    global_shap=shap_summaries.get('global', pd.DataFrame()),
    perm_importance=perm_summary,
    lead_importance=lead_importance,
    consensus_importance=consensus,
    top_k=15,
    save_path=f"{RESULTS_DIR}/clinical_report.txt"
)
print(report[:500])

# Step 6 — Error analysis
print("\n[6/6] Running error analysis...")

# Fix semua patient_id ke string — konsisten
metadata['patient_id']   = metadata['patient_id'].astype(str)
feature_df['patient_id'] = feature_df['patient_id'].astype(str)
fold_df['patient_id']    = fold_df['patient_id'].astype(str)

# Fix shap_result patient_ids juga
shap_result['patient_ids'] = shap_result['patient_ids'].astype(str)

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
print(error_results['narrative'][:1000])
print("\nInterpretability complete.")