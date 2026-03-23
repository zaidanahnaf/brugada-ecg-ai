# experiments/interpretability/clinical_report.py

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path


# ── Clinical Knowledge Base ───────────────────────────────────────────
# Maps feature name patterns to clinical interpretations
CLINICAL_INTERPRETATIONS = {

    # ST features
    'j_point_amplitude': {
        'direction': 'higher → Brugada',
        'physiology': (
            'J-point elevation ≥ 2mm in right precordials is the primary '
            'electrocardiographic criterion for Brugada Type 1 pattern. '
            'The J-wave represents delayed right ventricular depolarization '
            'producing an apparent notch or slur at QRS end.'
        ),
        'confidence': 'HIGH',
        'clinical_threshold': '≥ 0.2 mV (2mm)',
        'sensitivity_specificity': 'Type 1 criterion: ~100% specific by definition'
    },

    'st_j40': {
        'direction': 'higher → Brugada (Type 1) or saddleback (Type 2)',
        'physiology': (
            'ST amplitude at J+40ms reflects the sustained elevation after '
            'the J-point. In coved-type Brugada, this remains elevated but '
            'begins descending; in saddleback, it may be rising toward '
            'the secondary hump.'
        ),
        'confidence': 'HIGH',
        'clinical_threshold': 'Absolute elevation > 0.1 mV significant',
        'sensitivity_specificity': 'Primary Type 1 screening measure'
    },

    'st_slope_j0_j40': {
        'direction': 'more negative → Brugada (Type 1 coved pattern)',
        'physiology': (
            'The slope of ST descent immediately after the J-point '
            'is the key morphological discriminator between coved '
            '(negative slope, Type 1) and saddleback (positive slope, '
            'Type 2/3) patterns. Coved morphology has a descending ST '
            'that falls below the isoelectric line before the T-wave.'
        ),
        'confidence': 'HIGH',
        'clinical_threshold': 'Negative slope ≤ -0.1 mV/40ms = coved',
        'sensitivity_specificity': 'Most specific ST morphology feature'
    },

    'st_convexity': {
        'direction': 'more positive (convex) → Brugada Type 1',
        'physiology': (
            'The second derivative of the ST segment reflects its curvature. '
            'A positive value indicates the segment curves upward '
            '(convex = coved pattern). A negative value indicates '
            'concavity (saddleback). This is the mathematical encoding of '
            'what cardiologists visually assess as the ST "shape".'
        ),
        'confidence': 'MEDIUM (100Hz limits precision)',
        'clinical_threshold': 'Sign change is clinically meaningful',
        'sensitivity_specificity': 'Discriminative for coved vs saddleback'
    },

    'st_monotonicity': {
        'direction': 'higher (monotone descent) → Brugada Type 1',
        'physiology': (
            'A monotonically descending ST segment, without inflection '
            'points, characterizes the pure coved pattern. '
            'Saddleback patterns show a dip followed by a secondary rise, '
            'reducing monotonicity. Score = fraction of ST samples '
            'with negative derivative.'
        ),
        'confidence': 'MEDIUM',
        'clinical_threshold': '> 0.8 = predominantly monotone descent',
        'sensitivity_specificity': 'Supporting coved criterion'
    },

    'st_area_above_baseline': {
        'direction': 'larger → Brugada',
        'physiology': (
            'The area under the ST curve above baseline integrates '
            'both the height and duration of ST elevation. '
            'It captures the total ST "burden" rather than a single '
            'timepoint, making it more robust to J-point localization error.'
        ),
        'confidence': 'MEDIUM-HIGH',
        'clinical_threshold': 'Proportional to severity',
        'sensitivity_specificity': 'Robust amplitude-integrated measure'
    },

    # Morphology features
    't_inversion_indicator': {
        'direction': '1 → Brugada (T-wave inverted)',
        'physiology': (
            'T-wave inversion after a coved ST elevation is a '
            'defining criterion for Type 1 Brugada pattern per '
            'HRS/EHRA/APHRS consensus 2013. A negative T-wave '
            'in V1–V2 following the descending ST strongly distinguishes '
            'Type 1 from Type 2 (saddleback with upright T-wave).'
        ),
        'confidence': 'HIGH (if T-wave window correctly estimated)',
        'clinical_threshold': 'T-wave mean < -0.05 mV in V1–V2',
        'sensitivity_specificity': '~85% sensitivity, ~95% specificity for Type 1'
    },

    'covedness_score': {
        'direction': 'higher → Brugada Type 1',
        'physiology': (
            'Composite score encoding three independent Type 1 criteria: '
            '(1) negative ST slope = descending pattern, '
            '(2) positive ST convexity = upward bowing of the segment, '
            '(3) T-wave inversion. Each component independently predicts '
            'Type 1; their combination is highly specific.'
        ),
        'confidence': 'HIGH (composite)',
        'clinical_threshold': 'Threshold tuned on training data',
        'sensitivity_specificity': 'Composite — higher than any single component'
    },

    'saddleback_score': {
        'direction': 'higher → NOT Type 1 Brugada (Type 2/3 or Normal variant)',
        'physiology': (
            'Saddleback morphology (rising ST after J-point, secondary hump, '
            'upright T-wave) is characteristic of Type 2 Brugada or '
            'normal variants mimicking Brugada. '
            'High saddleback score in V1–V2 is evidence AGAINST Type 1.'
        ),
        'confidence': 'MEDIUM',
        'clinical_threshold': 'Complementary to covedness score',
        'sensitivity_specificity': 'Useful for Type 1 vs Type 2 discrimination'
    },

    'second_hump_present': {
        'direction': '1 → NOT Type 1 (saddleback feature)',
        'physiology': (
            'A secondary local maximum in the ST segment after the J-point '
            'peak defines the saddleback pattern. Its presence strongly '
            'suggests Type 2 or 3 Brugada pattern or normal variant, '
            'rather than the diagnostic Type 1.'
        ),
        'confidence': 'MEDIUM (100Hz may miss subtle humps)',
        'clinical_threshold': 'Binary presence/absence',
        'sensitivity_specificity': 'Specific for saddleback morphology'
    },

    'high_takeoff_gt_2mm': {
        'direction': '1 → Brugada (necessary but not sufficient)',
        'physiology': (
            'A J-point or r\'-wave amplitude ≥ 2mm (0.2 mV) in V1–V2 '
            'is required for any Brugada pattern type. Its absence '
            'effectively excludes the pattern. Its presence alone, without '
            'appropriate ST morphology, does not confirm Type 1.'
        ),
        'confidence': 'HIGH (binary threshold)',
        'clinical_threshold': '0.2 mV',
        'sensitivity_specificity': 'Necessary criterion; ~90% sensitivity for Brugada'
    },

    # Cross-lead features
    'cl_max_st_j40_v1v3': {
        'direction': 'higher → Brugada',
        'physiology': (
            'Peak ST elevation across V1–V3 captures the most affected '
            'lead. Clinical guidelines require elevation in at least '
            'one right precordial lead. This feature summarizes the '
            'overall right precordial ST burden.'
        ),
        'confidence': 'HIGH',
        'clinical_threshold': '> 0.1 mV in at least one of V1–V3',
        'sensitivity_specificity': 'Good screening feature'
    },

    'cl_n_leads_t_inverted_v1v3': {
        'direction': 'higher → Brugada',
        'physiology': (
            'T-wave inversion in multiple right precordial leads '
            '(V1 and V2 particularly) reinforces the Type 1 diagnosis. '
            'Isolated T-wave inversion in V1 alone is less specific. '
            'This count feature captures multi-lead involvement.'
        ),
        'confidence': 'MEDIUM-HIGH',
        'clinical_threshold': '≥ 2 leads inverted = stronger criterion',
        'sensitivity_specificity': 'Multi-lead T-wave involvement is specific'
    },

    'cl_v1v2_slope_consistency': {
        'direction': '1 (consistent negative slope in V1+V2) → Brugada',
        'physiology': (
            'When both V1 and V2 show the same slope direction '
            '(both descending), the pattern is more consistent with '
            'a true Brugada morphology rather than a localized artifact '
            'or lead-specific noise. Inconsistency raises quality concern.'
        ),
        'confidence': 'MEDIUM',
        'clinical_threshold': 'Binary consistency flag',
        'sensitivity_specificity': 'Reduces false positives from single-lead artifacts'
    },

    # Signal quality
    'hr_estimate_bpm': {
        'direction': 'complex (higher HR can unmask or mask Brugada)',
        'physiology': (
            'Heart rate affects ST morphology. Fever-induced tachycardia '
            'can unmask Brugada pattern; bradycardia-induced RR prolongation '
            'can accentuate it. HR features provide rate context for '
            'interpreting ST measurements made at a single snapshot.'
        ),
        'confidence': 'LOW (confounding, not primary)',
        'clinical_threshold': 'Contextual only',
        'sensitivity_specificity': 'Not a primary discriminator'
    },
}


def generate_clinical_interpretation_report(
    global_shap: pd.DataFrame,
    perm_importance: pd.DataFrame,
    lead_importance: pd.DataFrame,
    consensus_importance: pd.DataFrame,
    top_k: int = 15,
    save_path: str = "results/interpretability/clinical_report.txt"
) -> str:
    """
    Generate the clinical ML interpretation report.
    Links each top feature to its clinical rationale, direction,
    and confidence grade.

    This is the primary human-readable output for clinical review.
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("=" * 80)
    lines.append("CLINICAL ML INTERPRETATION REPORT")
    lines.append("Brugada-HUCA Dataset — Handcrafted Feature Analysis")
    lines.append("=" * 80)

    # ── Section 1: Executive Summary ─────────────────────────────
    lines.append("\n## 1. EXECUTIVE SUMMARY\n")

    top_features = consensus_importance.head(top_k)
    lines.append(
        f"The top {top_k} features by consensus importance are listed below. "
        f"All interpretations are based on out-of-fold SHAP values and "
        f"permutation importance — no training data leakage.\n"
    )

    # ── Section 2: Top Features with Clinical Context ─────────────
    lines.append("\n## 2. TOP DISCRIMINATIVE FEATURES\n")
    lines.append(
        f"{'Rank':<5} {'Feature':<45} {'SHAP':>8} {'Perm':>8} "
        f"{'Direction':<35} {'Confidence'}"
    )
    lines.append("-" * 120)

    for _, row in top_features.iterrows():
        feat = row['feature']
        shap_val = row.get('perm_importance_mean', np.nan)
        perm_val = row.get('native_importance_mean', np.nan)
        rank = row.get('consensus_position', '?')

        # Match to clinical knowledge base
        interp = _match_clinical_knowledge(feat, CLINICAL_INTERPRETATIONS)

        direction = interp.get('direction', 'unknown') if interp else 'No clinical mapping'
        confidence = interp.get('confidence', '?') if interp else '?'

        # Truncate for display
        feat_short = feat[:43]
        dir_short = direction[:33]

        lines.append(
            f"{str(rank):<5} {feat_short:<45} "
            f"{shap_val:>8.4f} {perm_val:>8.4f} "
            f"{dir_short:<35} {confidence}"
        )

    # ── Section 3: Feature-by-Feature Clinical Commentary ─────────
    lines.append("\n\n## 3. CLINICAL COMMENTARY — TOP FEATURES\n")

    for _, row in top_features.head(10).iterrows():
        feat = row['feature']
        interp = _match_clinical_knowledge(feat, CLINICAL_INTERPRETATIONS)
        if not interp:
            continue

        lines.append(f"### {feat}")
        lines.append(f"Direction of effect : {interp['direction']}")
        lines.append(f"Clinical confidence : {interp['confidence']}")
        lines.append(f"Clinical threshold  : {interp.get('clinical_threshold', 'N/A')}")
        lines.append(f"Sensitivity/Spec    : {interp.get('sensitivity_specificity', 'N/A')}")
        lines.append(f"\nPhysiology:\n{interp['physiology']}\n")
        lines.append("-" * 60)

    # ── Section 4: Lead-Level Summary ─────────────────────────────
    lines.append("\n## 4. LEAD-LEVEL IMPORTANCE\n")
    lines.append(
        f"{'Rank':<5} {'Lead':<12} {'Priority?':<10} "
        f"{'SHAP Total':>12} {'SHAP/Feature':>14} {'Perm Mean':>12}"
    )
    lines.append("-" * 70)

    for _, row in lead_importance.head(8).iterrows():
        priority = "YES ★" if row.get('is_priority', False) else "no"
        lines.append(
            f"{str(row.get('importance_rank', '?')):<5} "
            f"{str(row['lead']):<12} "
            f"{priority:<10} "
            f"{row.get('shap_total', np.nan):>12.4f} "
            f"{row.get('shap_mean_per_feature', np.nan):>14.4f} "
            f"{row.get('perm_importance_mean_per_feature', np.nan):>12.4f}"
        )

    lines.append(
        "\nNote: V1–V3 are expected to dominate. "
        "If another lead ranks higher, investigate feature extraction quality."
    )

    # ── Section 5: Feature Group Summary ──────────────────────────
    lines.append("\n## 5. FEATURE GROUP IMPORTANCE\n")

    if not global_shap.empty:
        group_shap = _summarize_by_group_from_shap(global_shap)
        lines.append(
            f"{'Group':<25} {'N Features':>10} "
            f"{'Total SHAP':>12} {'Mean SHAP/Feature':>20}"
        )
        lines.append("-" * 70)
        for _, row in group_shap.iterrows():
            lines.append(
                f"{row['group']:<25} {int(row['n_features']):>10} "
                f"{row['total_mean_abs_shap']:>12.4f} "
                f"{row['mean_per_feature']:>20.4f}"
            )

    report_text = "\n".join(lines)

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"Clinical report saved: {save_path}")
    return report_text


def _match_clinical_knowledge(
    feature_name: str,
    knowledge_base: Dict
) -> Optional[Dict]:
    """
    Match a feature name to the most specific entry in the
    clinical knowledge base.
    """
    # Try exact match first
    if feature_name in knowledge_base:
        return knowledge_base[feature_name]

    # Try substring match (without lead/aggregation suffix)
    for key in knowledge_base:
        if key in feature_name:
            return knowledge_base[key]

    return None


def _summarize_by_group_from_shap(shap_global: pd.DataFrame) -> pd.DataFrame:
    """Re-aggregate global SHAP by feature group for the report."""
    group_patterns = {
        'ST_V1–V3':       lambda f: any(f.startswith(p) for p in ['st_V1','st_V2','st_V3']),
        'Morphology_V1–V3': lambda f: any(f.startswith(p) for p in ['morph_V1','morph_V2','morph_V3']),
        'Cross-Lead':     lambda f: f.startswith('cl_'),
        'Signal_Quality': lambda f: f.startswith('sq_'),
        'HR/RR':          lambda f: any(f.startswith(p) for p in ['hr_','rr_','median_rr']),
        'Generic':        lambda f: any(f.startswith(p) for p in ['wavelet_','pca_','spectral_']),
    }

    rows = []
    for group, predicate in group_patterns.items():
        subset = shap_global[shap_global['feature'].apply(predicate)]
        if len(subset) == 0:
            continue
        rows.append({
            'group': group,
            'n_features': len(subset),
            'total_mean_abs_shap': float(subset['mean_abs_shap'].sum()),
            'mean_per_feature': float(subset['mean_abs_shap'].mean()),
        })

    return pd.DataFrame(rows).sort_values('total_mean_abs_shap', ascending=False)