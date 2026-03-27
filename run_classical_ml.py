# run_classical_ml.py

import logging
import pandas as pd
from src.config.__init__ import CFG
from src.fold_manager import load_folds
from src.feature_store import get_clean_feature_matrix
from experiments.model_registry import get_all_models
from experiments.models.classical_ml import run_cv_for_model
from experiments.results_table import build_comparison_table, print_comparison_table

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s — %(message)s'
)


def main():
    # ── Load Data ─────────────────────────────────────────────────
    feature_df = pd.read_csv("outputs/features/feature_matrix.csv")
    fold_df = load_folds("data/splits/fold_assignments.csv")

    target_meta_cols = [
        'patient_id', CFG.data.target_col,
        'pipeline_status', 'n_valid_beats', 'fold_id'
    ]
    feature_cols = [c for c in feature_df.columns if c not in target_meta_cols]

    # Drop high-missingness features
    feature_df_clean, feature_cols_clean = get_clean_feature_matrix(
        feature_df, feature_cols, max_missing_pct=50.0
    )

    print(f"Features after missingness filter: {len(feature_cols_clean)}")
    print(f"Subjects: {len(feature_df_clean)} | "
          f"Positive: {feature_df_clean[CFG.data.target_col].sum()}")

    # ── Experiment Grid ───────────────────────────────────────────
    # Vary imbalance strategy and feature selection independently
    experiment_configs = [
        # Primary: class_weight + univariate selection
        {'imbalance': 'class_weight',   'feat_sel': 'univariate_f',  'k': 40},
        # Compare: undersampling
        {'imbalance': 'undersample',    'feat_sel': 'univariate_f',  'k': 40},
        # Compare: no feature selection (all features)
        {'imbalance': 'class_weight',   'feat_sel': 'none',          'k': None},
        # Compare: L1-based selection
        {'imbalance': 'class_weight',   'feat_sel': 'l1_lr',         'k': 40},
    ]

    all_summaries = []
    models = get_all_models()

    # Primary experiment: all models with default config
    primary_cfg = experiment_configs[0]
    for model_cfg in models:
        summary = run_cv_for_model(
            model_config=model_cfg,
            feature_df=feature_df_clean,
            fold_df=fold_df,
            feature_cols=feature_cols_clean,
            imbalance_strategy=primary_cfg['imbalance'],
            feature_selection_method=primary_cfg['feat_sel'],
            k_features=primary_cfg['k'],
            threshold_method='youden',
            calibration_method='sigmoid',
            results_dir="outputs/results/primary"
        )
        all_summaries.append(summary)

    # Imbalance strategy comparison (best model only)
    # (extend to top-3 after primary results are in)
    best_model_name = sorted(
        all_summaries,
        key=lambda d: d.get('default_auroc_mean', 0),
        reverse=True
    )[0]['model_name']

    best_model_cfg = next(
        m for m in models if m['name'] == best_model_name
    )

    for exp_cfg in experiment_configs[1:]:
        summary = run_cv_for_model(
            model_config=best_model_cfg,
            feature_df=feature_df_clean,
            fold_df=fold_df,
            feature_cols=feature_cols_clean,
            imbalance_strategy=exp_cfg['imbalance'],
            feature_selection_method=exp_cfg['feat_sel'],
            k_features=exp_cfg.get('k', 40),
            threshold_method='youden',
            calibration_method='sigmoid',
            results_dir=f"outputs/results/imbalance_compare"
        )
        summary['experiment_variant'] = str(exp_cfg)
        all_summaries.append(summary)

    # ── Print Results ─────────────────────────────────────────────
    print_comparison_table(all_summaries)

    # Save master table
    comparison_df = build_comparison_table(all_summaries)
    comparison_df.to_csv("outputs/results/model_comparison_table.csv", index=False)
    print("Saved: outputs/results/model_comparison_table.csv")


if __name__ == "__main__":
    main()