# Brugada CNN Irzi Branch

## Project Structure

```
brugada_cnn/
├── src/
│   ├── config.py           ← ALL constants, paths, hyperparameters
│   ├── load_data.py        ← WFDB loader + fold SHA256 verification
│   ├── preprocess.py       ← Baseline removal, filtering, normalization
│   ├── beat_segmentation.py← R-peak detection, beat window extraction
│   ├── dataset.py          ← PyTorch Dataset classes
│   ├── model.py            ← 1D-CNN + ResNet-1D architectures
│   ├── train.py            ← 5-fold training loop
│   ├── evaluate.py         ← Metrics, beat aggregation, Youden threshold
│   └── integration_check.py← Pre-delivery validation
├── run_pipeline.py         ← MASTER SCRIPT — run this
└── requirements.txt
```

Expected project layout (relative to brugada_cnn/):
```
../data/raw/
    metadata.csv
    metadata_dictionary.csv
    files/
        188981/
            188981.dat
            188981.hea
        ... (362 more)
../data/splits/
    fold_assignments.csv    ← MUST EXIST before running
../features/                ← Created by pipeline
../results/                 ← Created by pipeline
../models/                  ← Created by pipeline
```

---

## Setup

```bash
pip install -r requirements.txt

# Verify GPU
python -c "import torch; print('GPU:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

# Quick smoke test (5 epochs, ~5 min)
python run_pipeline.py --smoke_test

# Full training — both 3-lead and 12-lead (ablation)
python run_pipeline.py

# Single variant
python run_pipeline.py --variant 3lead
python run_pipeline.py --variant 12lead
```

---

## What Gets Produced

After `run_pipeline.py` completes:

| File | Description |
|------|-------------|
| `features/cnn_fold_probs.csv` | 363 rows — OOF probability per subject |
| `features/cnn_embeddings.csv` | 363 rows — 64-dim OOF embedding per subject |
| `results/cnn_cv_summary.json` | Mean±std AUROC, AUPRC, Sensitivity, etc. |
| `models/cnn_*_fold_0..4.pt` | 5 saved model checkpoints |

Per-variant files (for ablation reference):
- `features/cnn_fold_probs_3lead.csv` / `cnn_fold_probs_12lead.csv`
- `features/cnn_embeddings_3lead.csv` / `cnn_embeddings_12lead.csv`
- `results/cnn_cv_summary_3lead.json` / `cnn_cv_summary_12lead.json`

---

## Critical Rules (enforced in code)

1. **Fold SHA256**: verified before every training run
   - Expected: `c8988aaa58ff7729ca36a7b54e7b6782f1ae2cd4e950dcf543fe120f11761904`
2. **No leakage**: `LeadNormalizer.fit()` called on training fold only
3. **Beat aggregation**: beat-level predictions -> `mean(beat_probs)` per subject
4. **Problematic subjects** (267630, 1230482): excluded from training, NaN in outputs
5. **patient_id**: always string, never integer
6. **OOF embeddings**: each subject's embedding from fold where they were in validation

---

## Architecture Decision (for technical report)

**Why 1D-CNN over 2D-CNN or Transformer?**
- Dataset size: 363 subjects is very small -> deep models will overfit
- 1D-CNN captures temporal patterns directly without 2D frequency encoding overhead
- Transformers need large datasets for self-attention to be meaningful
- Simple 1D-CNN with aggressive regularization (dropout 0.5, weight decay, early stopping) is optimal

**Why V1-V3 first?**
- Person 3 showed ST features in V1-V3 alone achieve AUROC 0.910
- Brugada Type 1 pattern is definitionally in right precordial leads
- Reduces model complexity -> less overfitting on small dataset
- Clinically interpretable: diagnostic leads are V1-V2

**Why beat-level training?**
- 363 recordings -> ~3,600 beats per fold (10× more samples)
- CNN needs sufficient samples; recording-level is risky with only 360
- Confirmed allowed by competition organizers (19 March)
- Predictions aggregated back to subject-level -> no evaluation leakage

**Why weighted BCE over SMOTE?**
- SMOTE generates synthetic samples -> explicitly forbidden
- Weighted BCE adjusts gradient importance without generating fake data
- pos_weight = 287/76 ≈ 3.77 compensates for 21% positive class ratio

---

## Hyperparameter Tuning

Default configuration (in `src/config.py`):
- `lr = 1e-4`
- `batch_size = 16`
- `epochs = 150` with early stopping patience=20
- `dropout = 0.5`
- `weight_decay = 1e-4`

To tune, edit `src/config.py` or pass CLI args:
```bash
python run_pipeline.py --lr 5e-4 --batch_size 8 --variant 12lead
```

---

## Minimum Performance Targets

| Metric | Target |
|--------|--------|
| AUROC | > 0.85 |
| AUPRC | > 0.70 |
| Sensitivity | > 0.75 (at Youden threshold) |

If AUROC < 0.80: revisit preprocessing (ST distortion?) or reduce model complexity.
If AUROC > 0.90: hybrid fusion with Person 3's CatBoost will be highly promising.

---

## For other team/people Baca Ini

Files to use:
- `features/cnn_fold_probs.csv` -> `oof_prob_brugada` column
- `features/cnn_embeddings.csv` -> `cnn_embed_0..63` columns
- Both are OOF-only (no leakage)
- `fold_id` in probs file matches `data/splits/fold_assignments.csv` exactly
