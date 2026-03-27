# scripts/read_ablation_conclusions.py

import pandas as pd
import numpy as np

df = pd.read_csv('outputs/results/ablation/ablation_master_table.csv')

def get(exp_id):
    row = df[df['exp_id'] == exp_id]
    if len(row) == 0 or pd.isna(row['AUROC'].iloc[0]):
        return np.nan, np.nan
    return float(row['AUROC'].iloc[0]), float(row['AUROC_std'].iloc[0])

def delta(exp_a, exp_b):
    a, _ = get(exp_a)
    b, _ = get(exp_b)
    return a - b if not (np.isnan(a) or np.isnan(b)) else np.nan

print("=" * 65)
print("MANDATORY ABLATION CONCLUSIONS")
print("=" * 65)

# Q1: Do ST features help?
st, st_s   = get('EXP_03_st_only_v1v3')
gen, gen_s = get('EXP_01_generic_stats_only')
d1 = delta('EXP_03_st_only_v1v3', 'EXP_01_generic_stats_only')
print(f"\nQ1: Do ST features help over generic stats?")
print(f"  Generic only  : {gen:.3f} ± {gen_s:.3f}")
print(f"  ST only V1-V3 : {st:.3f} ± {st_s:.3f}")
print(f"  Delta AUROC   : {d1:+.3f}")
print(f"  Conclusion    : {'YES — substantial improvement' if d1 >= 0.03 else 'marginal'}")

# Q2: Do morphology features help?
sm, sm_s = get('EXP_05_st_plus_morph_v1v3')
d2 = delta('EXP_05_st_plus_morph_v1v3', 'EXP_03_st_only_v1v3')
print(f"\nQ2: Do morphology features add over ST alone?")
print(f"  ST only       : {st:.3f} ± {st_s:.3f}")
print(f"  ST + Morph    : {sm:.3f} ± {sm_s:.3f}")
print(f"  Delta AUROC   : {d2:+.3f}")
print(f"  Conclusion    : {'YES' if d2 >= 0.01 else 'MARGINAL — ST alone captures most signal'}")

# Q3: Which leads matter most?
v1, v1s = get('EXP_11_v1_only')
v2, v2s = get('EXP_12_v2_only')
v12, v12s = get('EXP_13_v1v2_only')
v123, v123s = get('EXP_09_v1v3_only_all_feature_types')
nov, novs = get('EXP_10_no_v1v3_all_others')
print(f"\nQ3: Which leads matter most?")
print(f"  V1 only       : {v1:.3f} ± {v1s:.3f}")
print(f"  V2 only       : {v2:.3f} ± {v2s:.3f}  ← best single lead")
print(f"  V1+V2         : {v12:.3f} ± {v12s:.3f}")
print(f"  V1+V2+V3      : {v123:.3f} ± {v123s:.3f}")
print(f"  No V1-V3      : {nov:.3f} ± {novs:.3f}")
print(f"  V3 adds       : {v123-v12:+.3f} over V1+V2")
print(f"  Conclusion    : V2 is best single lead. V1+V2+V3 adds {v123-v1:+.3f} over V1 alone.")

# Q4: ST removal impact
all_hc, all_hc_s = get('EXP_07_all_handcrafted')
no_st, no_st_s   = get('EXP_14_all_except_st')
d4 = delta('EXP_07_all_handcrafted', 'EXP_14_all_except_st')
print(f"\nQ4: Impact of removing ST features?")
print(f"  All handcrafted   : {all_hc:.3f} ± {all_hc_s:.3f}")
print(f"  Without ST        : {no_st:.3f} ± {no_st_s:.3f}")
print(f"  Delta AUROC       : {d4:+.3f}")
print(f"  Conclusion        : {'ST features matter' if d4 >= 0.01 else 'Morphology compensates for ST'}")

# Q5: Morphology removal impact
no_morph, no_morph_s = get('EXP_15_all_except_morph')
d5 = delta('EXP_07_all_handcrafted', 'EXP_15_all_except_morph')
print(f"\nQ5: Impact of removing morphology features?")
print(f"  All handcrafted   : {all_hc:.3f} ± {all_hc_s:.3f}")
print(f"  Without morphology: {no_morph:.3f} ± {no_morph_s:.3f}")
print(f"  Delta AUROC       : {d5:+.3f}")
print(f"  Conclusion        : {'Morphology matters' if d5 >= 0.005 else 'ST alone carries most signal'}")

# Q6: Cross-lead impact
no_cl, no_cl_s = get('EXP_16_all_except_crosslead')
d6 = delta('EXP_07_all_handcrafted', 'EXP_16_all_except_crosslead')
print(f"\nQ6: Cross-lead feature contribution?")
print(f"  All handcrafted   : {all_hc:.3f} ± {all_hc_s:.3f}")
print(f"  Without crosslead : {no_cl:.3f} ± {no_cl_s:.3f}")
print(f"  Delta AUROC       : {d6:+.3f}")
print(f"  Conclusion        : {'Cross-lead adds signal' if d6 >= 0.005 else 'Marginal — individual lead features sufficient'}")

# Q7: T-wave contribution
no_tw, no_tw_s = get('EXP_17_all_except_twave')
d7 = delta('EXP_07_all_handcrafted', 'EXP_17_all_except_twave')
print(f"\nQ7: T-wave feature contribution?")
print(f"  All handcrafted   : {all_hc:.3f} ± {all_hc_s:.3f}")
print(f"  Without T-wave    : {no_tw:.3f} ± {no_tw_s:.3f}")
print(f"  Delta AUROC       : {d7:+.3f}")
print(f"  Conclusion        : {'T-wave adds signal' if d7 >= 0.005 else 'T-wave marginal at 100Hz — expected'}")

# Q8: Generic features add anything?
gen_all, gen_all_s = get('EXP_08_all_features_including_generic')
d8 = delta('EXP_08_all_features_including_generic', 'EXP_07_all_handcrafted')
print(f"\nQ8: Do generic features add over clinical features?")
print(f"  All handcrafted   : {all_hc:.3f} ± {all_hc_s:.3f}")
print(f"  + Generic features: {gen_all:.3f} ± {gen_all_s:.3f}")
print(f"  Delta AUROC       : {d8:+.3f}")
print(f"  Conclusion        : {'Generic features help' if d8 >= 0.005 else 'NO — clinical features sufficient'}")

# Q9: Stability summary
print(f"\nQ9: Are results stable across folds? (std < 0.06 = stable)")
key_exps = [
    'EXP_01_generic_stats_only',
    'EXP_03_st_only_v1v3',
    'EXP_07_all_handcrafted',
    'EXP_09_v1v3_only_all_feature_types',
]
for exp in key_exps:
    a, s = get(exp)
    status = 'STABLE' if s < 0.06 else 'UNSTABLE'
    print(f"  {exp:<45}: std={s:.3f} [{status}]")

print(f"\n{'='*65}")
print("SUMMARY TABLE")
print(f"{'='*65}")
print(f"{'Experiment':<45} {'AUROC':>8} {'±':>4} {'Rank'}")
print("-" * 65)
ranked = df.dropna(subset=['AUROC']).sort_values('AUROC', ascending=False)
for i, (_, row) in enumerate(ranked.iterrows(), 1):
    print(f"{row['exp_id']:<45} {row['AUROC']:>8.3f} {row['AUROC_std']:>6.3f}  #{i}")