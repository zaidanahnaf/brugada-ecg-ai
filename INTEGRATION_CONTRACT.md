
# HYBRID MODEL INTEGRATION CONTRACT
## Person 3 (Handcrafted Features) ↔ Person 2 (CNN)

---

## SECTION 1: SHARED GROUND TRUTH

FILE: data/splits/fold_assignments.csv
OWNER: Person 3
COLUMNS: patient_id, brugada, fold_id
FOLD RANGE: fold_id in {0, 1, 2, 3, 4}

RULE: This file MUST NOT be regenerated after initial creation.
      Both parties load this file from the shared repository root.

---

## SECTION 2: PERSON 3 DELIVERABLES TO PERSON 2

FILE 1: outputs/features/feature_matrix.csv
    Rows    : One per subject (363 total)
    Columns : patient_id, brugada, pipeline_status, n_valid_beats,
              [all handcrafted feature columns]
    Note    : Exclude pipeline_status == 'FAILED' rows before ML

FILE 2: outputs/features/feature_manifest.json
    Purpose : Complete feature registry with definitions and statistics

FILE 3: outputs/features/scaled/fold_{k}/
    Contents: train_X.npy, train_y.npy, train_ids.csv,
              val_X.npy, val_y.npy, val_ids.csv,
              imputer.pkl, scaler.pkl, feature_names.json
    Critical: DO NOT re-scale — use as-is

FILE 4: outputs/features/oof_predictions_handcrafted.csv
    Columns : patient_id, brugada, oof_prob_brugada, fold_id, model_name
    Critical: OOF only — no training predictions

FILE 5: outputs/features/reduced/top_k_{10,20,35}_feature_matrix.csv
    Note    : Use top_k_20 as default for early fusion

---

## SECTION 3: PERSON 2 DELIVERABLES TO PERSON 3

FILE 1: outputs/features/cnn_embeddings.csv
    Columns : patient_id, cnn_embed_0, ..., cnn_embed_{D-1}
    Critical: All 363 subjects — NaN for failures, do not drop rows

FILE 2: outputs/features/cnn_fold_probs.csv
    Columns : patient_id, brugada, oof_prob_brugada, fold_id
    Critical: Must use same fold_assignments.csv

FILE 3: outputs/results/cnn_cv_summary.json
    Contents: auroc_mean, auroc_std, auprc_mean, sensitivity_mean, etc.

---

## SECTION 4: FUSION EXPERIMENTS

Person 3 runs:
    EXP_20: CNN-only baseline
    EXP_21: CNN + all handcrafted (early fusion)
    EXP_22: CNN + ST+Morph only (early fusion)
    Late fusion: equal, auroc-weighted, optimized
    Stacking: logistic meta-learner

---

## SECTION 5: LEAKAGE PREVENTION CHECKLIST

[ ] fold_assignments.csv SHA256 matches agreed hash
[ ] No subject in both train and val of same fold
[ ] Scaler/imputer fitted ONLY on training data
[ ] CNN embedding extraction does NOT use val fold statistics
[ ] OOF predictions contain ONLY val-fold rows per subject
[ ] No metadata fields in primary features
[ ] Feature importance computed on OOF val data only

---

## SECTION 6: FAILURE HANDLING

IF CNN embeddings missing for some subjects:
    Fill with NaN — do not drop row
    Report count of imputed subjects

IF subject counts differ:
    Inner join on patient_id — report excluded subjects
    If >5% excluded: investigate before running fusion

IF fold assignments conflict:
    STOP — both parties reload from shared fold_assignments.csv
