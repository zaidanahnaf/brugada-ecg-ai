# CNN Branch — Brugada Syndrome ECG Classifier

## Person 2 Deliverables

---

## Overview

This directory contains the deep learning branch of the hybrid Brugada classifier.

**Task:** Binary classification — Normal (0) vs Brugada Syndrome (1 & 2)  
**Architecture:** 1D-CNN on V1/V2/V3 ECG leads (3-channel, 1200 time steps)  
**Dataset:** Brugada-HUCA (PhysioNet), 363 subjects (76 Brugada, 287 Normal)  
**Fold scheme:** 5-fold, defined by `data/splits/fold_assignments.csv` (SHA256-verified)

---

## Project Structure

```
brugada_cnn/
├── configs/
│   └── config.py              # All hyperparameters and paths (edit here only)
├── src/
│   ├── data_loader.py         # WFDB loading, fold assignment parsing
│   ├── preprocessing.py       # Baseline wander removal, LP filter, normalization
│   ├── dataset.py             # PyTorch Dataset + ST-safe augmentation
│   ├── model.py               # 1D-CNN with exposed 64-dim embedding
│   ├── trainer.py             # Training loop, early stopping, embedding extraction
│   ├── metrics.py             # AUROC, AUPRC, sensitivity/specificity, Brier
│   └── utils.py               # SHA256 check, seed setting, logging
├── scripts/
│   ├── train_cnn.py           # MAIN: 5-fold OOF training + deliverable generation
│   ├── tune_hyperparams.py    # Optional HP grid search (fold 0 only)
│   ├── qc_preprocessing.py    # Visual QC: raw vs preprocessed ECG plots
│   └── check_integration.py   # MANDATORY: integration checks before delivery
└── tests/
    └── test_components.py     # Unit tests for all components
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install torch wfdb neurokit2 numpy pandas scipy scikit-learn matplotlib tqdm
```

### 2. Run unit tests (verify everything before training)

```bash
python tests/test_components.py
```

### 3. Visual QC (verify ST morphology preserved)

```bash
python scripts/qc_preprocessing.py
# → Review results/qc_plots/*.png
```

### 4. (Optional) Hyperparameter tuning

```bash
python scripts/tune_hyperparams.py --leads 3
# → Results in results/hp_tuning_results.csv
```

### 5. Full 5-fold training (primary Model A — V1/V2/V3)

```bash
python scripts/train_cnn.py --leads 3
```

### 6. (Optional) 12-lead ablation

```bash
python scripts/train_cnn.py --leads 12
```

### 7. Integration checks (MANDATORY before delivery)

```bash
python scripts/check_integration.py
```

---

## Architecture

```
Input (batch, 3, 1200)              ← V1, V2, V3 leads at 100 Hz

Block 1: Conv1d(3→32, k=7) → BN → ReLU → MaxPool(/2)    → (batch, 32, 600)
Block 2: Conv1d(32→64, k=5) → BN → ReLU → MaxPool(/2)   → (batch, 64, 300)
Block 3: Conv1d(64→128, k=5) → BN → ReLU → MaxPool(/2)  → (batch, 128, 150)
Block 4: Conv1d(128→256, k=3) → BN → ReLU → AvgPool(1)  → (batch, 256, 1)

Flatten → (batch, 256)
FC: Linear(256→64) → ReLU → Dropout(0.5)               ← [PENULTIMATE EMBEDDING]
Output: Linear(64→1)                                     ← logits / sigmoid prob
```

- **~220K parameters** (3-lead) — appropriate for 363-subject dataset
- **64-dim embedding** — exposed for downstream fusion (Person 4 contract)
- Logits during training; sigmoid at inference

---

## Preprocessing Pipeline

All steps per-fold; normalization statistics **fit on training fold only**:

1. **Baseline wander removal** — median filter, kernel=61 samples (~610 ms)  
   _Not a high-pass filter — ST morphology is preserved_

2. **Low-pass filter** — Butterworth 40 Hz, order 4, zero-phase (`filtfilt`)  
   _Removes high-frequency noise; no phase distortion on ST segment_

3. **Per-lead amplitude normalization** — zero mean, unit std  
   _Statistics computed on training fold; applied identically to validation fold_

---

## Failed Subjects

The following subjects are **excluded from training** due to signal quality issues:

| patient_id | Reason                 | Handling                              |
| ---------- | ---------------------- | ------------------------------------- |
| 267630     | TOO_FEW_RPEAKS: 0      | NaN in embeddings and OOF probability |
| 1230482    | TOO_FEW_VALID_BEATS: 0 | NaN in embeddings and OOF probability |

Both are Brugada-positive. Their rows are preserved in all output CSVs with NaN values.

---

## Output Files

| File                                             | Shape    | Description                               |
| ------------------------------------------------ | -------- | ----------------------------------------- |
| `features/cnn_fold_probs.csv`                    | 363 × 4  | OOF probabilities, labels, fold IDs       |
| `features/cnn_embeddings.csv`                    | 363 × 65 | OOF 64-dim embeddings per subject         |
| `results/cnn_cv_summary.json`                    | —        | 5-fold metrics, best params, n_parameters |
| `models/cnn_fold_{0-4}.pt`                       | —        | Best checkpoint per fold                  |
| `preprocessing/fold_stats/fold_k_normalizer.npz` | —        | Per-fold normalization stats              |

---

## Minimum Performance Targets

| Metric           | Target | Status                       |
| ---------------- | ------ | ---------------------------- |
| AUROC mean       | > 0.85 | _(run training to populate)_ |
| AUPRC mean       | > 0.70 |                              |
| Sensitivity mean | > 0.75 |                              |

---

## Integration Contract (Person 4)

Your downstream fusion can use:

- `features/cnn_fold_probs.csv` → OOF probabilities for late fusion / stacking
- `features/cnn_embeddings.csv` → 64-dim embeddings for early fusion
- `features/oof_predictions_handcrafted.csv` → Person 3's OOF probabilities

All fold IDs are aligned with `data/splits/fold_assignments.csv`.  
All patient IDs are strings.

---

## Reproducibility

```
Python: 3.10+
PyTorch: 2.0+
Seed: 42 (global) + fold_id (per-fold)
Device: CPU
torch.backends.cudnn.deterministic = True
```

Full environment: `pip freeze > requirements_cnn.txt`
