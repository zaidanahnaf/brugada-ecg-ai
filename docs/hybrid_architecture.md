┌─────────────────────────────────────────────────────────────────────────┐
│ HYBRID FUSION ARCHITECTURE │
│ │
│ PERSON 3 OUTPUT PERSON 2 OUTPUT │
│ (This document) (CNN Engineer) │
│ │
│ feature_matrix.csv ←→ cnn_embeddings.csv │
│ feature_matrix_scaled/ cnn_logits.csv │
│ feature_manifest.json cnn_fold_probs.csv │
│ fold_assignments.csv ←→ fold_assignments.csv (SAME FILE) │
│ │
│ ↓ ↓ │
│ ┌─────────────────┐ ┌───────────────────────┐ │
│ │ Handcrafted │ │ CNN Branch │ │
│ │ Feature Branch │ │ (1D-CNN / ResNet) │ │
│ └────────┬────────┘ └──────────┬────────────┘ │
│ │ │ │
│ └──────────┬────────────────────┘ │
│ ↓ │
│ ┌─────────────────────┐ │
│ │ FUSION LAYER │ │
│ │ │ │
│ │ Strategy A: │ │
│ │ Early Fusion │ concat → meta-learner │
│ │ │ │
│ │ Strategy B: │ │
│ │ Late Fusion │ weighted avg of probabilities │
│ │ │ │
│ │ Strategy C: │ │
│ │ Stacking │ OOF probs → logistic meta-learner │
│ └─────────────────────┘ │
│ ↓ │
│ ┌─────────────────────┐ │
│ │ Final Prediction │ │
│ │ P(Brugada) │ │
│ └─────────────────────┘ │
│ │
│ INVARIANT: fold_assignments.csv is the single source of truth. │
│ Both branches use identical train/val splits in all experiments. │
└─────────────────────────────────────────────────────────────────────────┘
