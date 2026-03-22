┌─────────────────────────────────────────────────────────────────────┐
│ INTERPRETABILITY PIPELINE │
│ │
│ LAYER 1 — Global Feature Importance │
│ → Permutation importance (model-agnostic, leakage-safe) │
│ → Model-native importance (tree models) │
│ → L1 coefficient magnitude (linear models) │
│ → Aggregated rank across methods │
│ │
│ LAYER 2 — SHAP Analysis │
│ → Global SHAP summary (beeswarm + bar) │
│ → Lead-level SHAP aggregation │
│ → Feature group SHAP aggregation │
│ → Per-subject SHAP waterfall (selected cases) │
│ │
│ LAYER 3 — Lead-Level Importance │
│ → Importance summed per lead │
│ → V1 vs V2 vs V3 contribution ranking │
│ → Cross-lead consistency analysis │
│ │
│ LAYER 4 — Clinical Interpretation Report │
│ → Top features linked to Brugada physiology │
│ → Direction of effect (higher X → more Brugada) │
│ → Confidence grading per feature │
│ │
│ LAYER 5 — Error Analysis │
│ → False negative profile (missed Brugada) │
│ → False positive profile (misclassified Normal) │
│ → Failure mode hypothesis generation │
│ │
│ ALL layers use ONLY out-of-fold predictions │
│ NO model is re-examined on its training data │
└─────────────────────────────────────────────────────────────────────┘
