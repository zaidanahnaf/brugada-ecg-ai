# experiments/interpretability/error_analysis.py

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path

from src.config import CFG


def run_error_analysis(
    feature_df: pd.DataFrame,
    shap_result: Dict,
    metadata: pd.DataFrame,
    model_factory,
    fold_df: pd.DataFrame,
    feature_cols: List[str],
    threshold: float = 0.5,
    results_dir: str = "results/interpretability"
) -> Dict[str, pd.DataFrame]:
    """
    Comprehensive error analysis.
    Profiles false negatives and false positives on out-of-fold predictions.

    Returns dict with:
        fn_profile    : DataFrame of false negative cases
        fp_profile    : DataFrame of false positive cases
        fn_vs_tp      : Feature value comparison FN vs TP (among Brugada)
        fp_vs_tn      : Feature value comparison FP vs TN (among Normal)
        error_summary : Narrative summary dict
    """
    from experiments.interpretability.shap_plots import _get_oof_probabilities
    from experiments.calibration import calibrate_model

    Path(results_dir).mkdir(parents=True, exist_ok=True)

    y_oof = shap_result['y_oof']
    patient_ids = shap_result['patient_ids']

    y_prob = _get_oof_probabilities(
        shap_result, feature_df, model_factory, fold_df, feature_cols
    )
    y_pred = (y_prob >= threshold).astype(int)

    # ── Case Classification ────────────────────────────────────────
    case_type = np.where(
        (y_oof == 1) & (y_pred == 1), 'TP',
        np.where(
            (y_oof == 1) & (y_pred == 0), 'FN',
            np.where(
                (y_oof == 0) & (y_pred == 1), 'FP', 'TN'
            )
        )
    )

    # Build case DataFrame
    case_df = pd.DataFrame({
        'patient_id': patient_ids,
        'true_label': y_oof,
        'predicted_label': y_pred,
        'predicted_prob': y_prob,
        'case_type': case_type
    })

    # Merge with features
    case_df = case_df.merge(
        feature_df[['patient_id'] + feature_cols].copy(),
        on='patient_id', how='left'
    )

    # Merge with metadata
    if metadata is not None:
        case_df = case_df.merge(
            metadata[['patient_id','basal_pattern','sudden_death']],
            on='patient_id', how='left'
        )

    # ── False Negative Profile ─────────────────────────────────────
    fn_df = case_df[case_df['case_type'] == 'FN'].copy()
    tp_df = case_df[case_df['case_type'] == 'TP'].copy()
    fp_df = case_df[case_df['case_type'] == 'FP'].copy()
    tn_df = case_df[case_df['case_type'] == 'TN'].copy()

    fn_vs_tp = _compare_groups(fn_df, tp_df, feature_cols, 'FN', 'TP')
    fp_vs_tn = _compare_groups(fp_df, tn_df, feature_cols, 'FP', 'TN')

    # ── Metadata Pattern Analysis ──────────────────────────────────
    metadata_analysis = {}
    if 'basal_pattern' in case_df.columns:
        metadata_analysis['fn_basal_pattern'] = fn_df['basal_pattern'].value_counts().to_dict()
        metadata_analysis['tp_basal_pattern'] = tp_df['basal_pattern'].value_counts().to_dict()
        metadata_analysis['fp_basal_pattern'] = fp_df['basal_pattern'].value_counts().to_dict()

    # ── Clinical Feature Profiles ──────────────────────────────────
    priority_features = [
        f for f in feature_cols
        if any(k in f for k in [
            'j_point_amplitude_median', 'st_slope_j0_j40_median',
            'st_j40_median', 't_inversion_indicator_median',
            'covedness_score_median', 'st_convexity_median'
        ])
    ]

    fn_profile = _build_case_profile(fn_df, priority_features, 'FN')
    fp_profile = _build_case_profile(fp_df, priority_features, 'FP')
    tp_profile = _build_case_profile(tp_df, priority_features, 'TP')

    # ── Error Summary Narrative ────────────────────────────────────
    error_summary = _generate_error_narrative(
        fn_df, fp_df, tp_df, tn_df,
        fn_vs_tp, fp_vs_tn,
        metadata_analysis
    )

    # Save
    fn_vs_tp.to_csv(f"{results_dir}/fn_vs_tp_feature_comparison.csv", index=False)
    fp_vs_tn.to_csv(f"{results_dir}/fp_vs_tn_feature_comparison.csv", index=False)
    fn_profile.to_csv(f"{results_dir}/fn_case_profiles.csv", index=False)
    fp_profile.to_csv(f"{results_dir}/fp_case_profiles.csv", index=False)

    with open(f"{results_dir}/error_analysis_narrative.txt", 'w') as f:
        f.write(error_summary)

    print(f"Error analysis saved to: {results_dir}/")
    print(f"  FN: {len(fn_df)} | FP: {len(fp_df)} | TP: {len(tp_df)} | TN: {len(tn_df)}")

    return {
        'fn_profile': fn_profile,
        'fp_profile': fp_profile,
        'tp_profile': tp_profile,
        'fn_vs_tp': fn_vs_tp,
        'fp_vs_tn': fp_vs_tn,
        'case_df': case_df,
        'narrative': error_summary
    }


def _compare_groups(
    group_a: pd.DataFrame,
    group_b: pd.DataFrame,
    feature_cols: List[str],
    name_a: str,
    name_b: str
) -> pd.DataFrame:
    """
    Compare feature distributions between two case groups.
    Reports mean, std, and effect size (Cohen's d) per feature.
    """
    rows = []
    for feat in feature_cols:
        if feat not in group_a.columns:
            continue
        vals_a = group_a[feat].dropna().values.astype(float)
        vals_b = group_b[feat].dropna().values.astype(float)

        if len(vals_a) < 2 or len(vals_b) < 2:
            continue

        mean_a = float(np.mean(vals_a))
        mean_b = float(np.mean(vals_b))
        std_a = float(np.std(vals_a))
        std_b = float(np.std(vals_b))

        pooled_std = np.sqrt(
            (std_a**2 * (len(vals_a)-1) + std_b**2 * (len(vals_b)-1)) /
            max(len(vals_a) + len(vals_b) - 2, 1)
        )
        cohen_d = (mean_a - mean_b) / max(pooled_std, 1e-9)

        rows.append({
            'feature': feat,
            f'mean_{name_a}': mean_a,
            f'mean_{name_b}': mean_b,
            f'std_{name_a}': std_a,
            f'std_{name_b}': std_b,
            'mean_difference': mean_a - mean_b,
            'cohen_d': float(cohen_d),
            'abs_cohen_d': abs(float(cohen_d))
        })

    df = pd.DataFrame(rows)
    return df.sort_values('abs_cohen_d', ascending=False)


def _build_case_profile(
    case_df: pd.DataFrame,
    priority_features: List[str],
    case_type: str
) -> pd.DataFrame:
    """Build a per-subject feature profile for a case group."""
    available = [f for f in priority_features if f in case_df.columns]
    if not available:
        return pd.DataFrame()

    profile = case_df[['patient_id', 'predicted_prob'] + available].copy()
    profile['case_type'] = case_type
    return profile.sort_values('predicted_prob')


def _generate_error_narrative(
    fn_df, fp_df, tp_df, tn_df,
    fn_vs_tp, fp_vs_tn,
    metadata_analysis
) -> str:
    """
    Generate the structured narrative error analysis report.
    """
    lines = []
    lines.append("=" * 80)
    lines.append("ERROR ANALYSIS REPORT")
    lines.append("=" * 80)

    # ── Confusion Matrix Summary ───────────────────────────────────
    total = len(fn_df) + len(fp_df) + len(tp_df) + len(tn_df)
    lines.append(f"\n## CONFUSION MATRIX SUMMARY (default threshold 0.5)\n")
    lines.append(f"  True Positives  (TP): {len(tp_df):>4}  Correctly identified Brugada")
    lines.append(f"  False Negatives (FN): {len(fn_df):>4}  Missed Brugada ← CLINICAL PRIORITY")
    lines.append(f"  False Positives (FP): {len(fp_df):>4}  Misclassified Normal")
    lines.append(f"  True Negatives  (TN): {len(tn_df):>4}  Correctly identified Normal")
    lines.append(f"  Total: {total}")

    if total > 0:
        lines.append(f"\n  Sensitivity: {len(tp_df)/max(len(tp_df)+len(fn_df),1):.3f}")
        lines.append(f"  Specificity: {len(tn_df)/max(len(tn_df)+len(fp_df),1):.3f}")

    # ── False Negative Analysis ────────────────────────────────────
    lines.append(f"\n## FALSE NEGATIVE ANALYSIS ({len(fn_df)} cases)\n")
    lines.append(
        "False negatives are Brugada patients predicted as Normal. "
        "These represent the highest clinical risk — missed diagnosis "
        "of Brugada syndrome can lead to sudden cardiac death."
    )

    if len(fn_df) > 0:
        lines.append(f"\n  Mean predicted probability: {fn_df['predicted_prob'].mean():.3f}")
        lines.append(f"  (TP mean: {tp_df['predicted_prob'].mean():.3f} for comparison)")

        # Metadata pattern
        if 'basal_pattern' in fn_df.columns:
            fn_basal = fn_df['basal_pattern'].value_counts()
            tp_basal = tp_df['basal_pattern'].value_counts()
            lines.append(f"\n  FN basal_pattern distribution: {fn_basal.to_dict()}")
            lines.append(f"  TP basal_pattern distribution: {tp_basal.to_dict()}")

            spontaneous_fn = fn_basal.get(1, 0)
            spontaneous_total_brugada = (fn_df['basal_pattern'].fillna(0) == 1).sum() + \
                                         (tp_df['basal_pattern'].fillna(0) == 1).sum()
            if spontaneous_total_brugada > 0:
                lines.append(
                    f"\n  ⚠ {spontaneous_fn} of {len(fn_df)} FN cases "
                    f"have spontaneous (basal) Type 1 pattern. "
                    + (
                        "This suggests the model struggles even with spontaneous patterns — "
                        "a serious concern." if spontaneous_fn > 0 else
                        "FN cases appear to be drug-induced/borderline patterns — "
                        "expected difficulty at 100 Hz with ambiguous morphology."
                    )
                )

        # Top features distinguishing FN from TP
        if not fn_vs_tp.empty:
            top_diff = fn_vs_tp.head(5)
            lines.append(f"\n  Top features where FN differ from TP:")
            for _, row in top_diff.iterrows():
                lines.append(
                    f"    {row['feature']:<45}: "
                    f"FN={row.get('mean_FN', np.nan):.3f} vs "
                    f"TP={row.get('mean_TP', np.nan):.3f} "
                    f"(d={row['cohen_d']:.2f})"
                )

    # Hypotheses for FN
    lines.append(f"\n  Hypotheses for false negatives:")
    lines.append(
        "  H1: Drug-induced / borderline patterns with submaximal ST elevation\n"
        "      → ECG at baseline may not show Type 1; only visible post-sodium-channel-blocker\n"
        "  H2: Low J-point at 100 Hz sampling: true <2mm elevation misquantified\n"
        "      → 100 Hz gives ±0.01 mV precision per sample; borderline cases ambiguous\n"
        "  H3: Spontaneous Type 1 pattern not present in the 12-second snapshot\n"
        "      → Brugada pattern is dynamic; single recording may miss it\n"
        "  H4: Poor beat quality in this subject → fiducial detection unreliable\n"
        "      → Check n_valid_beats and j_point_confidence flags for FN cases"
    )

    # ── False Positive Analysis ────────────────────────────────────
    lines.append(f"\n## FALSE POSITIVE ANALYSIS ({len(fp_df)} cases)\n")
    lines.append(
        "False positives are Normal patients predicted as Brugada. "
        "While less dangerous than FN, these cause unnecessary testing "
        "and patient anxiety."
    )

    if len(fp_df) > 0:
        lines.append(f"\n  Mean predicted probability: {fp_df['predicted_prob'].mean():.3f}")

        if not fp_vs_tn.empty:
            top_diff_fp = fp_vs_tn.head(5)
            lines.append(f"\n  Top features where FP differ from TN:")
            for _, row in top_diff_fp.iterrows():
                lines.append(
                    f"    {row['feature']:<45}: "
                    f"FP={row.get('mean_FP', np.nan):.3f} vs "
                    f"TN={row.get('mean_TN', np.nan):.3f} "
                    f"(d={row['cohen_d']:.2f})"
                )

    lines.append(f"\n  Hypotheses for false positives:")
    lines.append(
        "  H1: Right bundle branch block (RBBB) or incomplete RBBB\n"
        "      → Produces rSR' pattern in V1 that mimics Brugada high-takeoff\n"
        "  H2: Early repolarization syndrome in young athletes\n"
        "      → ST elevation in right precordials without Brugada morphology\n"
        "  H3: Electrode misplacement (V1/V2 too high)\n"
        "      → Intercostal space placement error produces Brugada-like pattern\n"
        "  H4: Noisy right precordial leads → spurious ST elevation artifact\n"
        "      → Check snr_proxy and hf_noise features for FP cases\n"
        "  H5: Acute myocardial infarction (Brugada phenocopy)\n"
        "      → ST elevation from ischemia, not sodium channelopathy"
    )

    # ── Pipeline Quality Check ────────────────────────────────────
    lines.append(f"\n## PIPELINE QUALITY CROSS-CHECK\n")
    lines.append(
        "Recommended checks for all error cases:\n"
        "  1. Verify n_valid_beats ≥ 5 (low beat count = unreliable features)\n"
        "  2. Check j_point_confidence flags (LOW_CONFIDENCE cases → noisy features)\n"
        "  3. Inspect snr_proxy for V1/V2/V3 (< 2.0 = questionable signal quality)\n"
        "  4. Compare pipeline_status: are errors clustered in PARTIAL status subjects?\n"
        "  5. For FN: inspect raw ECG waveform manually for any visible Brugada pattern"
    )

    lines.append("\n" + "=" * 80)
    return "\n".join(lines)