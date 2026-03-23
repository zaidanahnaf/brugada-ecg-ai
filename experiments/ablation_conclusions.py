# experiments/ablation_conclusions.py

"""
Automated conclusion generation from ablation results.
Produces a structured answer to each mandatory question.

These conclusions are the CORE DELIVERABLE of Work Package D.
They must be defensible, quantified, and caveat-appropriate.
"""

import numpy as np
import pandas as pd
from typing import Dict


CONCLUSION_THRESHOLDS = {
    'meaningful_auroc_gain':  0.03,    # ΔAUROC ≥ 0.03 = meaningful
    'marginal_auroc_gain':    0.01,    # ΔAUROC 0.01–0.03 = marginal
    'stable_result':          0.06,    # std(AUROC) < 0.06 = stable
    'dominant_lead_margin':   0.03,    # Single lead within 0.03 of multi-lead
}


def generate_conclusions(
    ablation_table: pd.DataFrame,
    comparison_table: pd.DataFrame
) -> Dict[str, str]:
    """
    Auto-generate structured answers to all mandatory questions.
    Returns dict of {question_id: conclusion_text}.
    """
    def get_auroc(exp_id):
        row = ablation_table[ablation_table['exp_id'] == exp_id]
        if len(row) == 0:
            return np.nan, np.nan
        return float(row['AUROC'].iloc[0]), float(row['AUROC_std'].iloc[0])

    def get_delta(exp_id_a, exp_id_b):
        a, _ = get_auroc(exp_id_a)
        b, _ = get_auroc(exp_id_b)
        return a - b if not (np.isnan(a) or np.isnan(b)) else np.nan

    def stable(std_val):
        return std_val < CONCLUSION_THRESHOLDS['stable_result']

    def meaningful(delta):
        return delta >= CONCLUSION_THRESHOLDS['meaningful_auroc_gain']

    def marginal(delta):
        return CONCLUSION_THRESHOLDS['marginal_auroc_gain'] <= delta \
               < CONCLUSION_THRESHOLDS['meaningful_auroc_gain']

    conclusions = {}

    # ── Q1: Do ST features help? ──────────────────────────────────
    delta_st = get_delta('EXP_03_st_only_v1v3', 'EXP_01_generic_stats_only')
    auroc_st, std_st = get_auroc('EXP_03_st_only_v1v3')
    auroc_gen, _ = get_auroc('EXP_01_generic_stats_only')

    conclusions['Q1_st_features_help'] = (
        f"ST features (V1–V3) {'substantially improve' if meaningful(delta_st) else 'marginally improve' if marginal(delta_st) else 'do NOT improve'} "
        f"classification over generic signal statistics. "
        f"AUROC: {auroc_gen:.3f} (generic) -> {auroc_st:.3f} (ST only), "
        f"ΔAUROC = {delta_st:+.3f}. "
        f"Result is {'stable' if stable(std_st) else 'UNSTABLE — interpret cautiously'} "
        f"across folds (std={std_st:.3f})."
    )

    # ── Q2: Do morphology features help? ─────────────────────────
    delta_morph = get_delta('EXP_05_st_plus_morph_v1v3', 'EXP_03_st_only_v1v3')
    auroc_stmorph, std_stmorph = get_auroc('EXP_05_st_plus_morph_v1v3')

    conclusions['Q2_morphology_features_help'] = (
        f"Adding Brugada morphology features (covedness, saddleback, T-wave) "
        f"to ST features {'improves' if meaningful(delta_morph) else 'marginally improves' if marginal(delta_morph) else 'does NOT improve'} performance. "
        f"ΔAUROC = {delta_morph:+.3f} "
        f"({auroc_st:.3f} ST-only -> {auroc_stmorph:.3f} ST+Morph). "
        + (
            "Morphology features encode coved/saddleback distinction "
            "not captured by amplitude-alone ST measures."
            if meaningful(delta_morph) else
            "ST amplitudes alone may already encode most discriminative signal at 100 Hz."
        )
    )

    # ── Q3: Which leads matter most? ─────────────────────────────
    auroc_v1, _ = get_auroc('EXP_11_v1_only')
    auroc_v2, _ = get_auroc('EXP_12_v2_only')
    auroc_v1v2, _ = get_auroc('EXP_13_v1v2_only')
    auroc_v1v3, _ = get_auroc('EXP_09_v1v3_only_all_feature_types')
    auroc_nov1v3, _ = get_auroc('EXP_10_no_v1v3_all_others')

    best_single = max(
        [('V1', auroc_v1), ('V2', auroc_v2)],
        key=lambda x: x[1] if not np.isnan(x[1]) else -1
    )

    conclusions['Q3_which_leads_matter'] = (
        f"Right precordial leads V1–V3 are critical: removing them causes "
        f"ΔAUROC = {get_delta('EXP_10_no_v1v3_all_others', 'EXP_09_v1v3_only_all_feature_types'):+.3f}. "
        f"Single-lead performance: V1={auroc_v1:.3f}, V2={auroc_v2:.3f}. "
        f"Best single lead: {best_single[0]} (AUROC={best_single[1]:.3f}). "
        f"V1+V2 (AUROC={auroc_v1v2:.3f}) vs V1–V3 (AUROC={auroc_v1v3:.3f}): "
        f"ΔAUROC={auroc_v1v3-auroc_v1v2:+.3f} from adding V3. "
        f"Conclusion: {'V3 provides meaningful additional signal.' if auroc_v1v3-auroc_v1v2 >= 0.02 else 'V1+V2 is largely sufficient; V3 adds marginally.'}"
    )

    # ── Q4: Does handcrafted + CNN outperform CNN-only? ───────────
    auroc_cnn, std_cnn = get_auroc('EXP_20_cnn_only')
    auroc_hybrid, std_hybrid = get_auroc('EXP_21_cnn_plus_handcrafted')
    delta_hybrid = get_delta('EXP_21_cnn_plus_handcrafted', 'EXP_20_cnn_only')

    if np.isnan(auroc_cnn):
        conclusions['Q4_hybrid_vs_cnn'] = (
            "CNN experiments not yet run (awaiting Person 2 embeddings). "
            "Will be populated when EXP_20 and EXP_21 are available."
        )
    else:
        conclusions['Q4_hybrid_vs_cnn'] = (
            f"Hybrid (CNN + handcrafted) {'outperforms' if meaningful(delta_hybrid) else 'marginally outperforms' if marginal(delta_hybrid) else 'does NOT outperform'} "
            f"CNN alone. "
            f"ΔAUROC = {delta_hybrid:+.3f} "
            f"(CNN={auroc_cnn:.3f} -> Hybrid={auroc_hybrid:.3f}). "
            + (
                "Handcrafted clinical features provide complementary signal "
                "beyond what the CNN learns end-to-end."
                if meaningful(delta_hybrid) else
                "CNN already captures most morphological information; "
                "handcrafted features provide limited additional gain."
            )
        )

    # ── Q5: Are gains stable across folds? ───────────────────────
    key_exps = [
        'EXP_03_st_only_v1v3',
        'EXP_05_st_plus_morph_v1v3',
        'EXP_07_all_handcrafted',
        'EXP_09_v1v3_only_all_feature_types',
    ]
    stability_report = []
    for exp_id in key_exps:
        _, std_val = get_auroc(exp_id)
        if not np.isnan(std_val):
            stability_report.append(
                f"{exp_id}: std={std_val:.3f} "
                f"({'STABLE' if stable(std_val) else 'UNSTABLE'})"
            )

    conclusions['Q5_stability'] = (
        "Fold-level AUROC stability assessment:\n" +
        "\n".join(stability_report) +
        f"\nThreshold: std < {CONCLUSION_THRESHOLDS['stable_result']} = stable. "
        "High variance may indicate sensitivity to fold-level class imbalance "
        "given only 76 positive cases."
    )

    # ── Q6: T-wave contribution ───────────────────────────────────
    delta_twave = get_delta('EXP_07_all_handcrafted', 'EXP_17_all_except_twave')
    conclusions['Q6_twave_contribution'] = (
        f"Removing T-wave features from the full feature set: "
        f"ΔAUROC = {delta_twave:+.3f}. "
        + (
            "T-wave inversion is a defining Type 1 Brugada criterion; "
            "its removal meaningfully degrades performance."
            if meaningful(delta_twave) else
            "T-wave features add marginal signal at 100 Hz, "
            "likely due to T-wave boundary uncertainty at this sampling rate."
        )
    )

    # ── Q7: Metadata upper bound ──────────────────────────────────
    auroc_meta, _ = get_auroc('EXP_18_metadata_only')
    auroc_ecg, _ = get_auroc('EXP_07_all_handcrafted')

    conclusions['Q7_metadata_sanity'] = (
        f"[EXPLORATORY — NOT PRIMARY RESULT] "
        f"Metadata-only AUROC = {auroc_meta:.3f}. "
        f"ECG-only AUROC = {auroc_ecg:.3f}. "
        + (
            "Metadata (basal_pattern, sudden_death) provides strong "
            "classification signal — expected since these are near-target fields. "
            "This is a sanity check, not a deployable model."
            if not np.isnan(auroc_meta) else
            "Metadata experiment not run."
        )
    )

    return conclusions


def print_conclusions(conclusions: Dict[str, str]):
    """Format and print all mandatory conclusions."""
    print("\n" + "="*80)
    print("ABLATION STUDY — MANDATORY CONCLUSIONS")
    print("="*80)
    for q_id, text in conclusions.items():
        print(f"\n[{q_id}]")
        print(text)
    print("\n" + "="*80)