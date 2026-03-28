"""
scripts/analyze_top_features.py
Feature importance analysis untuk CatBoost + LogReg + RF
"""

import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# =============================================================================
# CONFIG
# =============================================================================
FOLD_CSV     = 'data/splits/fold_assignments.csv'
FEATURE_CSV  = 'features/feature_matrix.csv'
MANIFEST     = 'features/catboost_feature_names.json'
FAILED_SUBJ  = {'267630', '1230482'}
SEED         = 42
TOP_N        = 20
OUT_DIR      = 'results/feature_importance'

import os; os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
# LOAD DATA
# =============================================================================
df_folds = pd.read_csv(FOLD_CSV, dtype={'patient_id': str})
df_feat  = pd.read_csv(FEATURE_CSV, dtype={'patient_id': str})

with open(MANIFEST) as f:
    manifest = json.load(f)
feature_cols = manifest['feature_names']

df = df_folds[['patient_id','brugada','fold_id']].merge(
    df_feat.drop(columns=['brugada'], errors='ignore'),
    on='patient_id', how='left'
)

df_clean = df[~df['patient_id'].isin(FAILED_SUBJ)].reset_index(drop=True)
y     = df_clean['brugada'].values.astype(int)
folds = df_clean['fold_id'].values.astype(int)
X_raw = df_clean[feature_cols].values.astype(np.float32)

# =============================================================================
# TRAIN ON ALL DATA (untuk importance — bukan OOF)
# Ini HANYA untuk visualisasi, bukan untuk prediksi
# =============================================================================
print("Training models on full data for importance analysis...")

imputer = SimpleImputer(strategy='median')
scaler  = StandardScaler()
X_imp   = imputer.fit_transform(X_raw)
X_all   = scaler.fit_transform(X_imp)

# CatBoost
cb = CatBoostClassifier(
    iterations=500, learning_rate=0.05, depth=4,
    l2_leaf_reg=5, auto_class_weights='Balanced',
    random_seed=SEED, verbose=0
)
cb.fit(X_all, y)
cb_importance = cb.get_feature_importance()

# LogReg (koefisien absolut)
lr = LogisticRegression(C=0.1, class_weight='balanced',
                         solver='saga', penalty='l1',
                         max_iter=2000, random_state=SEED)
lr.fit(X_all, y)
lr_importance = np.abs(lr.coef_[0])

# RF
rf = RandomForestClassifier(n_estimators=300, max_depth=6,
                              min_samples_leaf=3,
                              class_weight='balanced',
                              random_state=SEED, n_jobs=-1)
rf.fit(X_all, y)
rf_importance = rf.feature_importances_

# =============================================================================
# RANK & PRINT TOP N
# =============================================================================

def get_top_n(importance, feature_names, n=TOP_N):
    idx = np.argsort(importance)[::-1][:n]
    return pd.DataFrame({
        'rank':       range(1, n+1),
        'feature':    [feature_names[i] for i in idx],
        'importance': [importance[i] for i in idx],
        'pct':        [100 * importance[i] / importance.sum() for i in idx],
    })

df_cb = get_top_n(cb_importance, feature_cols)
df_lr = get_top_n(lr_importance, feature_cols)
df_rf = get_top_n(rf_importance, feature_cols)

print(f"\n{'='*65}")
print(f"  TOP {TOP_N} FEATURES — CATBOOST")
print(f"{'='*65}")
for _, row in df_cb.iterrows():
    bar = '█' * int(row['pct'] / 2)
    print(f"  {int(row['rank']):>2}. {row['feature']:<45} {row['pct']:>5.1f}%  {bar}")

print(f"\n{'='*65}")
print(f"  TOP {TOP_N} FEATURES — LOGREG (|coef|)")
print(f"{'='*65}")
for _, row in df_lr.iterrows():
    bar = '█' * int(row['pct'] / 2)
    print(f"  {int(row['rank']):>2}. {row['feature']:<45} {row['pct']:>5.1f}%  {bar}")

print(f"\n{'='*65}")
print(f"  TOP {TOP_N} FEATURES — RANDOM FOREST")
print(f"{'='*65}")
for _, row in df_rf.iterrows():
    bar = '█' * int(row['pct'] / 2)
    print(f"  {int(row['rank']):>2}. {row['feature']:<45} {row['pct']:>5.1f}%  {bar}")

# =============================================================================
# CONSENSUS RANKING — rata-rata rank dari 3 model
# =============================================================================
all_features = set(feature_cols)
ranks = {f: [] for f in all_features}

for df_imp in [df_cb, df_lr, df_rf]:
    for _, row in df_imp.iterrows():
        ranks[row['feature']].append(row['rank'])

# Features yang tidak masuk top N diberi rank = N+1
for f in ranks:
    while len(ranks[f]) < 3:
        ranks[f].append(TOP_N + 1)

consensus = pd.DataFrame([
    {'feature': f, 'mean_rank': np.mean(v), 'in_all_3': len([x for x in v if x <= TOP_N]) == 3}
    for f, v in ranks.items()
]).sort_values('mean_rank').head(TOP_N).reset_index(drop=True)
consensus['rank'] = range(1, len(consensus)+1)

print(f"\n{'='*65}")
print(f"  TOP {TOP_N} CONSENSUS RANKING (rata-rata rank dari 3 model)")
print(f"{'='*65}")
for _, row in consensus.iterrows():
    tag = " ★ semua model" if row['in_all_3'] else ""
    print(f"  {int(row['rank']):>2}. {row['feature']:<45} mean_rank={row['mean_rank']:.1f}{tag}")

# =============================================================================
# CEK DOMINANSI st_V1_st_slope_j20_j80_min
# =============================================================================
target_feat = 'st_V1_st_slope_j20_j80_min'
print(f"\n{'='*65}")
print(f"  DOMINANSI: {target_feat}")
print(f"{'='*65}")

for name, df_imp in [('CatBoost', df_cb), ('LogReg', df_lr), ('RF', df_rf)]:
    row = df_imp[df_imp['feature'] == target_feat]
    if len(row) > 0:
        r = row.iloc[0]
        print(f"  {name:<12} rank=#{int(r['rank']):<3} importance={r['pct']:.2f}%")
    else:
        print(f"  {name:<12} tidak masuk top {TOP_N}")

cb_rank = df_cb[df_cb['feature']==target_feat]['rank'].values
if len(cb_rank) > 0 and cb_rank[0] == 1:
    pct = df_cb[df_cb['feature']==target_feat]['pct'].values[0]
    runner_up_pct = df_cb.iloc[1]['pct']
    print(f"\n  → Dominansi vs #2: {pct:.1f}% vs {runner_up_pct:.1f}% "
          f"(ratio {pct/runner_up_pct:.1f}x)")

# =============================================================================
# PLOT — horizontal bar chart top 20 CatBoost
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 8))
fig.suptitle('Feature Importance — Brugada ECG Classifier', fontsize=13, fontweight='bold')

colors = {'st_': '#e74c3c', 'morph_': '#e67e22', 'sq_': '#95a5a6'}

def get_color(fname):
    for prefix, color in colors.items():
        if fname.startswith(prefix):
            return color
    return '#3498db'

for ax, (title, df_imp) in zip(axes, [
    ('CatBoost', df_cb), ('LogReg |coef|', df_lr), ('Random Forest', df_rf)
]):
    top = df_imp.head(TOP_N).iloc[::-1]   # reverse untuk plot bawah ke atas
    bar_colors = [get_color(f) for f in top['feature']]
    bars = ax.barh(range(len(top)), top['pct'], color=bar_colors, alpha=0.85)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([f[:40] for f in top['feature']], fontsize=7)
    ax.set_xlabel('Importance (%)')
    ax.set_title(title, fontweight='bold')
    ax.axvline(x=top['pct'].mean(), color='gray', linestyle='--', alpha=0.5, label='mean')

    # Highlight target feature
    for i, fname in enumerate(top['feature']):
        if fname == target_feat:
            ax.get_yticklabels()[i].set_color('red')
            ax.get_yticklabels()[i].set_fontweight('bold')

# Legend
from matplotlib.patches import Patch
legend = [
    Patch(color='#e74c3c', label='ST features'),
    Patch(color='#e67e22', label='Morph features'),
    Patch(color='#95a5a6', label='Signal quality'),
    Patch(color='#3498db', label='Other'),
]
axes[0].legend(handles=legend, loc='lower right', fontsize=8)

plt.tight_layout()
plot_path = f'{OUT_DIR}/feature_importance_top{TOP_N}.png'
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n[OK] Plot saved: {plot_path}")

# =============================================================================
# SAVE CSV
# =============================================================================
df_cb.to_csv(f'{OUT_DIR}/importance_catboost.csv', index=False)
df_lr.to_csv(f'{OUT_DIR}/importance_logreg.csv', index=False)
df_rf.to_csv(f'{OUT_DIR}/importance_rf.csv', index=False)
consensus.to_csv(f'{OUT_DIR}/importance_consensus.csv', index=False)
print(f"[OK] CSVs saved: {OUT_DIR}/")