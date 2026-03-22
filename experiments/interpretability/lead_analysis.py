# experiments/interpretability/lead_analysis.py

import numpy as np
import pandas as pd
from typing import Dict, List


def build_lead_importance_report(
    perm_summary: pd.DataFrame,
    shap_summaries: Dict,
    feature_cols: List[str]
) -> pd.DataFrame:
    """
    Unified lead-level importance combining permutation and SHAP.
    """
    leads = ['V1','V2','V3','V4','V5','V6',
             'I','II','III','aVR','aVL','aVF','cross_lead']

    rows = []
    for lead in leads:
        # Features belonging to this lead
        if lead == 'cross_lead':
            lead_feats = [f for f in feature_cols if 'cl_' in f]
        else:
            lead_feats = [
                f for f in feature_cols
                if f'_{lead}_' in f or f.startswith(f'st_{lead}_')
                or f.startswith(f'morph_{lead}_')
            ]

        if not lead_feats:
            continue

        # Permutation importance for this lead
        lead_perm = perm_summary[
            perm_summary['feature'].isin(lead_feats)
        ]
        mean_perm = float(
            lead_perm['perm_importance_mean'].mean()
        ) if len(lead_perm) > 0 else np.nan
        sum_perm = float(
            lead_perm['perm_importance_mean'].sum()
        ) if len(lead_perm) > 0 else np.nan

        # SHAP importance for this lead
        shap_global = shap_summaries.get('global', pd.DataFrame())
        if not shap_global.empty:
            lead_shap = shap_global[
                shap_global['feature'].isin(lead_feats)
            ]
            mean_shap = float(
                lead_shap['mean_abs_shap'].mean()
            ) if len(lead_shap) > 0 else np.nan
            sum_shap = float(
                lead_shap['mean_abs_shap'].sum()
            ) if len(lead_shap) > 0 else np.nan
        else:
            mean_shap = sum_shap = np.nan

        rows.append({
            'lead': lead,
            'n_features': len(lead_feats),
            'is_priority': lead in ['V1','V2','V3'],
            'perm_importance_mean_per_feature': mean_perm,
            'perm_importance_total': sum_perm,
            'shap_mean_per_feature': mean_shap,
            'shap_total': sum_shap,
            'consensus_importance': np.nanmean([mean_perm, mean_shap])
        })

    df = pd.DataFrame(rows).sort_values(
        'consensus_importance', ascending=False
    )
    df['importance_rank'] = range(1, len(df) + 1)
    return df


def analyze_v1v2v3_contributions(
    shap_result: Dict,
    feature_cols: List[str]
) -> pd.DataFrame:
    """
    Detailed V1 vs V2 vs V3 contribution analysis.
    For each feature TYPE (ST amplitude, slope, covedness, etc.),
    compare importance across V1, V2, V3.
    """
    shap_vals = shap_result['shap_values']
    feat_names = shap_result['feature_names']

    # Feature types present across leads
    feature_types = set()
    for f in feat_names:
        for lead in ['V1','V2','V3']:
            stub = f.replace(f'_{lead}_', '_LEAD_')
            if stub != f:
                feature_types.add(stub)

    rows = []
    for ftype in sorted(feature_types):
        row = {'feature_type': ftype}
        for lead in ['V1','V2','V3']:
            target = ftype.replace('_LEAD_', f'_{lead}_')
            # Find all matching features (including aggregation suffixes)
            matching = [
                i for i, f in enumerate(feat_names)
                if f.startswith(target) or f == target
            ]
            if matching:
                # Use mean across aggregation stats (median, mean, std)
                lead_shap = np.abs(shap_vals[:, matching]).mean()
                row[f'shap_{lead}'] = float(lead_shap)
            else:
                row[f'shap_{lead}'] = np.nan

        # Which lead dominates for this feature type?
        shap_vals_leads = [row.get(f'shap_{l}', np.nan) for l in ['V1','V2','V3']]
        valid_vals = [(l, v) for l, v in zip(['V1','V2','V3'], shap_vals_leads)
                      if not np.isnan(v)]
        if valid_vals:
            row['dominant_lead'] = max(valid_vals, key=lambda x: x[1])[0]
            row['v1v2v3_variance'] = float(
                np.var([v for _, v in valid_vals])
            )
        else:
            row['dominant_lead'] = 'unknown'
            row['v1v2v3_variance'] = np.nan

        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        'shap_V2', ascending=False, na_position='last'
    )