# scripts/verify_all_inputs.py

import pandas as pd
import numpy as np
import hashlib
import json
from pathlib import Path

print("=" * 60)
print("PRE-FLIGHT VERIFICATION")
print("=" * 60)

checks_passed = 0
checks_total  = 0

def check(name, condition, detail=""):
    global checks_passed, checks_total
    checks_total += 1
    if condition:
        checks_passed += 1
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")

# 1. Fold file SHA256
with open("data/splits/fold_assignments.csv", 'rb') as f:
    sha = hashlib.sha256(f.read()).hexdigest()
expected_sha = "c8988aaa58ff7729ca36a7b54e7b6782f1ae2cd4e950dcf543fe120f11761904"
check("Fold SHA256", sha == expected_sha, f"got {sha[:16]}...")

# 2. Feature matrix
feat_df = pd.read_csv("features/feature_matrix.csv")
feat_df['patient_id'] = feat_df['patient_id'].astype(str)
check("Feature matrix rows",    len(feat_df) == 363)
check("Feature matrix positive", feat_df['brugada'].sum() == 76)
check("Feature matrix no metadata leak",
      'basal_pattern' not in feat_df.columns and
      'sudden_death'  not in feat_df.columns)

# 3. OOF handcrafted predictions
hc_oof = pd.read_csv("features/oof_predictions_handcrafted.csv")
hc_oof['patient_id'] = hc_oof['patient_id'].astype(str)
check("HC OOF rows",       len(hc_oof) == 363)
check("HC OOF prob range", hc_oof['oof_prob_brugada'].between(0,1).all())

# 4. CNN files (from Person 2)
cnn_emb_path  = Path("features/cnn_embeddings.csv")
cnn_prob_path = Path("features/cnn_fold_probs.csv")

if cnn_emb_path.exists() and cnn_prob_path.exists():
    cnn_emb  = pd.read_csv(cnn_emb_path)
    cnn_prob = pd.read_csv(cnn_prob_path)
    cnn_emb['patient_id']  = cnn_emb['patient_id'].astype(str)
    cnn_prob['patient_id'] = cnn_prob['patient_id'].astype(str)

    embed_cols = [c for c in cnn_emb.columns if c.startswith('cnn_embed_')]
    fold_df    = pd.read_csv("data/splits/fold_assignments.csv")
    fold_df['patient_id'] = fold_df['patient_id'].astype(str)

    merged    = cnn_prob.merge(fold_df[['patient_id','fold_id']],
                               on='patient_id', suffixes=('_cnn','_ref'))
    mismatch  = (merged['fold_id_cnn'] != merged['fold_id_ref']).sum()

    check("CNN embeddings rows",   len(cnn_emb) == 363)
    check("CNN probs rows",        len(cnn_prob) == 363)
    check("CNN embed dims",        len(embed_cols) == 64)
    check("CNN prob range",        cnn_prob['oof_prob_brugada'].between(0,1).all())
    check("CNN fold alignment",    mismatch == 0,
          f"{mismatch} fold_id mismatches")
    check("CNN label consistency", (cnn_prob['brugada'] ==
          hc_oof.set_index('patient_id').loc[
              cnn_prob['patient_id'].values, 'brugada'].values).all())
else:
    print("  SKIP  CNN files not yet available — waiting for Person 2")

print(f"\n{checks_passed}/{checks_total} checks passed")
if checks_passed == checks_total:
    print("ALL CHECKS PASSED — Safe to proceed")
else:
    print("FIX FAILURES BEFORE PROCEEDING")