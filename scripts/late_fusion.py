# scripts/late_fusion.py

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

df_cnn    = pd.read_csv('features/cnn_fold_probs.csv',
                         dtype={'patient_id': str}).dropna(subset=['oof_prob_brugada'])
df_logreg = pd.read_csv('features/oof_predictions_handcrafted.csv',
                         dtype={'patient_id': str})

df = df_cnn.merge(df_logreg[['patient_id','oof_prob_brugada']]
                  .rename(columns={'oof_prob_brugada':'prob_logreg'}),
                  on='patient_id')
df = df.rename(columns={'oof_prob_brugada': 'prob_cnn'})

y          = df['brugada'].values
prob_cnn   = df['prob_cnn'].values
prob_logreg = df['prob_logreg'].values

print("Grid search weighted average:")
print(f"{'w_cnn':>8} {'w_logreg':>10} {'AUROC':>8}")
print("-" * 30)

best_auroc, best_w = 0, 0
for w_cnn in np.arange(0.0, 1.05, 0.05):
    w_logreg  = 1.0 - w_cnn
    prob_fused = w_cnn * prob_cnn + w_logreg * prob_logreg
    auroc      = roc_auc_score(y, prob_fused)
    marker     = " ←" if auroc > best_auroc else ""
    print(f"{w_cnn:>8.2f} {w_logreg:>10.2f} {auroc:>8.4f}{marker}")
    if auroc > best_auroc:
        best_auroc, best_w = auroc, w_cnn

print(f"\nBest: w_cnn={best_w:.2f}, AUROC={best_auroc:.4f}")