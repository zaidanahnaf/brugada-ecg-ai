┌─────────────────────────────────────────────────────────────────┐
│ OUTER LOOP (Evaluation) │
│ 5-Fold Stratified CV — fold_assignments.csv │
│ │
│ ┌───────────────────────────────────────────────────────────┐ │
│ │ INNER LOOP (Hyperparameter Search) │ │
│ │ 4-Fold CV on training portion of outer fold │ │
│ │ GridSearch / RandomSearch per model │ │
│ │ Best params selected by AUROC on inner val │ │
│ └───────────────────────────────────────────────────────────┘ │
│ │
│ Per outer fold: │
│ 1. Fit imputer + scaler on X_train │
│ 2. Fit feature selector on X_train │
│ 3. Fit model with best inner-loop params on X_train │
│ 4. Evaluate on X_val (held-out outer fold) │
│ 5. Tune threshold on X_val using Youden index │
│ 6. Calibrate model on X_val (isotonic / Platt) │
│ 7. Record all metrics │
│ │
│ Final result = mean ± std of metrics across 5 outer folds │
└─────────────────────────────────────────────────────────────────┘
