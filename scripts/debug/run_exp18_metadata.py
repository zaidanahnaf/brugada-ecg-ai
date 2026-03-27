# scripts/run_exp18_metadata.py — fix lengkap

import pandas as pd
import logging
from src.config.__init__ import CFG
from src.fold_manager import load_folds
from src.data.data_loader import load_metadata
from experiments.ablation import run_single_ablation_experiment
from catboost import CatBoostClassifier

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s — %(message)s')

feature_df = pd.read_csv("outputs/features/feature_matrix.csv")
fold_df    = load_folds("data/splits/fold_assignments.csv")
metadata   = load_metadata(CFG.data.metadata_path)

# Fix semua patient_id ke string — konsisten di semua DataFrame
feature_df['patient_id'] = feature_df['patient_id'].astype(str)
metadata['patient_id']   = metadata['patient_id'].astype(str)
fold_df['patient_id']    = fold_df['patient_id'].astype(str)   # ← ini yang kurang

# Merge metadata columns
feature_df_meta = feature_df.merge(
    metadata[['patient_id', 'basal_pattern', 'sudden_death']],
    on='patient_id', how='left'
)

# Verifikasi alignment
print(f"feature_df_meta rows  : {len(feature_df_meta)}")
print(f"fold_df rows          : {len(fold_df)}")
print(f"basal_pattern values  : {feature_df_meta['basal_pattern'].value_counts().to_dict()}")
print(f"sudden_death values   : {feature_df_meta['sudden_death'].value_counts().to_dict()}")

# Quick alignment check
common_ids = set(feature_df_meta['patient_id']) & set(fold_df['patient_id'])
print(f"IDs matched           : {len(common_ids)} / {len(fold_df)}")

def model_factory():
    return CatBoostClassifier(
        learning_rate=0.1, l2_leaf_reg=3,
        iterations=100, depth=3, border_count=32,
        auto_class_weights='Balanced',
        eval_metric='AUC', random_seed=42, verbose=0
    )

result = run_single_ablation_experiment(
    exp_id='EXP_18_metadata_only',
    feature_subset=['basal_pattern', 'sudden_death'],
    feature_df=feature_df_meta,
    fold_df=fold_df,
    model_factory=model_factory,
    results_dir="outputs/results/ablation"
)

print(f"\nEXP_18 Metadata Only:")
if result.get('status') == 'OK':
    print(f"  AUROC : {result.get('default_auroc_mean', 0):.3f} ± {result.get('default_auroc_std', 0):.3f}")
    print(f"  AUPRC : {result.get('default_auprc_mean', 0):.3f}")
    print(f"  Note  : Exploratory only — near-target features, NOT primary result")
else:
    print(f"  Status: {result.get('status')}")
    print(f"  Error : {result.get('failure_reason', 'unknown')}")