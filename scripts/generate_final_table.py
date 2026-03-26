# scripts/generate_final_table.py

import pandas as pd
import numpy as np
import json
import glob

rows = []

# Classical ML results
classical = pd.read_csv("results/model_comparison_table.csv")
for _, row in classical.iterrows():
    rows.append({
        'Method'    : row['Model'],
        'Type'      : 'Classical ML',
        'AUROC'     : row['AUROC'],
        'Sensitivity': row['Sensitivity'],
        'Specificity': row['Specificity'],
    })

# CNN result
with open("results/cnn_cv_summary.json") as f:
    cnn = json.load(f)
rows.append({
    'Method'    : 'CNN (1D-ResNet)',
    'Type'      : 'Deep Learning',
    'AUROC'     : f"{cnn['auroc_mean']:.3f} ± {cnn['auroc_std']:.3f}",
    'Sensitivity': f"{cnn['sensitivity_mean']:.3f}",
    'Specificity': f"{cnn['specificity_mean']:.3f}",
})

# Hybrid results
for strategy in ['early_fusion', 'late_fusion', 'stacking']:
    result_path = f"results/hybrid/{strategy}/"
    files = glob.glob(f"{result_path}*.json")
    if files:
        with open(files[0]) as f:
            h = json.load(f)
        rows.append({
            'Method'    : f"Hybrid ({strategy.replace('_',' ').title()})",
            'Type'      : 'Hybrid',
            'AUROC'     : f"{h.get('default_auroc_mean',0):.3f} ± {h.get('default_auroc_std',0):.3f}",
            'Sensitivity': f"{h.get('tuned_sensitivity_mean',0):.3f}",
            'Specificity': f"{h.get('tuned_specificity_mean',0):.3f}",
        })

df = pd.DataFrame(rows)
df.to_csv("results/final_comparison_table.csv", index=False)
print(df.to_string(index=False))