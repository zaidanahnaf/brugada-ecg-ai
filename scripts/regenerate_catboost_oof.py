# scripts/regenerate_catboost_oof.py — versi fix

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from src.config import CFG
from src.fold_manager import load_folds
from src.feature_store import get_clean_feature_matrix, fit_scaler_imputer, transform

fold_df    = load_folds("data/splits/fold_assignments.csv")
feature_df = pd.read_csv("features/feature_matrix.csv")
feature_df['patient_id'] = feature_df['patient_id'].astype(str)
fold_df['patient_id']    = fold_df['patient_id'].astype(str)

# Ground truth labels — SELALU dari fold_df, bukan feature_df
label_map = dict(zip(fold_df['patient_id'], fold_df['brugada']))

feature_cols = [
    c for c in feature_df.columns
    if c not in ['patient_id', CFG.target_col,
                 'pipeline_status', 'n_valid_beats']
    and not c.startswith('qc_')
]

df_clean, feat_clean = get_clean_feature_matrix(
    feature_df, feature_cols, max_missing_pct=50.0)

all_rows = []

for val_fold in range(5):
    val_ids_fold   = fold_df[fold_df['fold_id'] == val_fold]['patient_id'].values
    train_ids_fold = fold_df[fold_df['fold_id'] != val_fold]['patient_id'].values

    # Ambil feature matrix rows berdasarkan patient_id
    train_df = df_clean[df_clean['patient_id'].isin(train_ids_fold)]
    val_df   = df_clean[df_clean['patient_id'].isin(val_ids_fold)]

    X_train = train_df[feat_clean].values.astype(np.float32)
    X_val   = val_df[feat_clean].values.astype(np.float32)

    # Labels DARI fold_df — ground truth yang benar
    y_train = np.array([label_map[pid] for pid in train_df['patient_id']])
    y_val   = np.array([label_map[pid] for pid in val_df['patient_id']])
    val_pids = val_df['patient_id'].values

    imputer, scaler = fit_scaler_imputer(X_train)
    X_train_proc    = transform(X_train, imputer, scaler)
    X_val_proc      = transform(X_val,   imputer, scaler)

    model = CatBoostClassifier(
        learning_rate=0.1, l2_leaf_reg=3,
        iterations=100, depth=3, border_count=32,
        auto_class_weights='Balanced',
        eval_metric='AUC', random_seed=42, verbose=0
    )
    model.fit(X_train_proc, y_train)
    probs = model.predict_proba(X_val_proc)[:, 1]

    fold_auc = roc_auc_score(y_val, probs)
    print(f"Fold {val_fold}: AUROC={fold_auc:.4f} | "
          f"n_pos={y_val.sum()} | n_val={len(y_val)}")

    for pid, prob, label in zip(val_pids, probs, y_val):
        all_rows.append({
            'patient_id':       str(pid),
            'brugada':          int(label),
            'oof_prob_brugada':  float(prob),
            'fold_id':          int(val_fold),
            'model_name':       'CatBoost_best'
        })

df_oof = pd.DataFrame(all_rows)
df_oof.to_csv("features/oof_predictions_catboost_best.csv", index=False)

# Verifikasi final dengan gt labels
final_auc = roc_auc_score(df_oof['brugada'], df_oof['oof_prob_brugada'])
print(f"\nOverall OOF AUROC (CatBoost best): {final_auc:.4f}")
print(f"Positive count: {df_oof['brugada'].sum()} (expected ~76)")
print(f"File saved: features/oof_predictions_catboost_best.csv")