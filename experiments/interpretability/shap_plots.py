# experiments/interpretability/shap_plots.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
from pathlib import Path
from typing import Dict, List, Optional

from src.experiments.calibration import calibrate_model

warnings.filterwarnings('ignore')


def plot_shap_beeswarm(
    shap_result: Dict,
    top_k: int = 25,
    save_path: str = "results/interpretability/shap_beeswarm.png"
):
    """
    SHAP beeswarm plot for top-K features.
    Color = feature value (red = high, blue = low).
    X-axis = SHAP value (right = pushes toward Brugada prediction).
    """
    try:
        import shap
    except ImportError:
        print("shap not installed")
        return

    shap_vals = shap_result['shap_values']
    X_oof = shap_result['X_oof']
    feat_names = shap_result['feature_names']

    # Select top-K by mean |SHAP|
    mean_abs = np.abs(shap_vals).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:top_k]

    fig, ax = plt.subplots(figsize=(12, max(8, top_k * 0.4)))
    shap.summary_plot(
        shap_vals[:, top_idx],
        X_oof[:, top_idx],
        feature_names=[feat_names[i] for i in top_idx],
        show=False,
        plot_type='dot',
        max_display=top_k,
        color_bar=True,
        plot_size=None,
        alpha=0.6
    )
    plt.title(
        f"SHAP Beeswarm — Top {top_k} Features\n"
        f"(Out-of-fold, n={len(shap_vals)} subjects)",
        fontsize=13, pad=12
    )
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_shap_bar(
    global_summary: pd.DataFrame,
    top_k: int = 20,
    save_path: str = "results/interpretability/shap_bar.png"
):
    """
    Horizontal bar chart of mean |SHAP| for top features.
    Color-coded by feature group.
    """
    GROUP_COLORS = {
        'st_V1': '#e63946',   'st_V2': '#e63946',  'st_V3': '#e63946',
        'morph_V1': '#2a9d8f','morph_V2': '#2a9d8f','morph_V3': '#2a9d8f',
        'cl_': '#457b9d',
        'sq_': '#a8dadc',
        'hr_': '#a8dadc',     'rr_': '#a8dadc',
    }

    def get_color(feat_name):
        for prefix, color in GROUP_COLORS.items():
            if feat_name.startswith(prefix) or prefix in feat_name:
                return color
        return '#cccccc'

    top = global_summary.head(top_k).copy()
    top = top.sort_values('mean_abs_shap', ascending=True)

    colors = [get_color(f) for f in top['feature']]

    fig, ax = plt.subplots(figsize=(10, max(6, top_k * 0.4)))
    bars = ax.barh(
        top['feature'], top['mean_abs_shap'],
        color=colors, edgecolor='white', linewidth=0.5
    )

    # Error bars using std
    if 'std_shap' in top.columns:
        ax.barh(
            top['feature'], top['std_shap'],
            left=top['mean_abs_shap'] - top['std_shap'],
            color='none', edgecolor='#333333',
            linewidth=0.8, height=0.1
        )

    ax.set_xlabel("Mean |SHAP Value|", fontsize=11)
    ax.set_title(
        f"Feature Importance — Mean |SHAP| (Top {top_k})",
        fontsize=13, pad=10
    )

    # Legend
    legend_elements = [
        mpatches.Patch(color='#e63946', label='ST Features (V1–V3)'),
        mpatches.Patch(color='#2a9d8f', label='Morphology (V1–V3)'),
        mpatches.Patch(color='#457b9d', label='Cross-Lead'),
        mpatches.Patch(color='#a8dadc', label='Signal Quality / HR'),
        mpatches.Patch(color='#cccccc', label='Other'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_lead_importance(
    lead_summary: pd.DataFrame,
    save_path: str = "results/interpretability/lead_importance.png"
):
    """
    Bar chart of SHAP importance aggregated by lead.
    Highlights V1–V3 relative to all other leads.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: total SHAP per lead
    ax = axes[0]
    colors = [
        '#e63946' if l in ['V1','V2','V3']
        else '#457b9d' if l == 'cl_'
        else '#adb5bd'
        for l in lead_summary['lead']
    ]
    bars = ax.bar(
        lead_summary['lead'],
        lead_summary['total_mean_abs_shap'],
        color=colors, edgecolor='white'
    )
    ax.set_title("Total Mean |SHAP| by Lead", fontsize=12)
    ax.set_xlabel("Lead")
    ax.set_ylabel("Sum of Mean |SHAP|")
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)

    # Right: per-feature SHAP by lead (normalized by feature count)
    ax2 = axes[1]
    colors2 = [
        '#e63946' if l in ['V1','V2','V3']
        else '#457b9d' if l == 'cl_'
        else '#adb5bd'
        for l in lead_summary['lead']
    ]
    ax2.bar(
        lead_summary['lead'],
        lead_summary['mean_abs_shap_per_feature'],
        color=colors2, edgecolor='white'
    )
    ax2.set_title("Mean |SHAP| per Feature by Lead\n(Normalized)", fontsize=12)
    ax2.set_xlabel("Lead")
    ax2.set_ylabel("Mean |SHAP| per feature")
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)

    # Shared legend
    legend_elements = [
        mpatches.Patch(color='#e63946', label='Priority (V1–V3)'),
        mpatches.Patch(color='#457b9d', label='Cross-lead'),
        mpatches.Patch(color='#adb5bd', label='Other leads'),
    ]
    fig.legend(
        handles=legend_elements,
        loc='upper center', ncol=3, fontsize=10,
        bbox_to_anchor=(0.5, 1.02)
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_shap_waterfall_cases(
    shap_result: Dict,
    feature_df: pd.DataFrame,
    case_type: str = 'fn',        # 'fn' | 'fp' | 'tp' | 'tn'
    n_cases: int = 3,
    model_factory=None,
    fold_df=None,
    feature_cols=None,
    save_dir: str = "results/interpretability/waterfall"
):
    """
    SHAP waterfall plots for selected individual cases.
    Most informative for false negatives (missed Brugada)
    and false positives (misclassified Normal).

    case_type:
        'fn' -> false negatives (Brugada predicted as Normal)
        'fp' -> false positives (Normal predicted as Brugada)
        'tp' -> true positives (correctly identified Brugada)
        'tn' -> true negatives (correctly identified Normal)
    """
    try:
        import shap
    except ImportError:
        print("shap not installed")
        return

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    shap_vals = shap_result['shap_values']
    y_oof = shap_result['y_oof']
    y_prob_oof = _get_oof_probabilities(
        shap_result, feature_df, model_factory, fold_df, feature_cols
    )
    feat_names = shap_result['feature_names']
    patient_ids = shap_result['patient_ids']

    threshold = 0.5   # Use default for case selection

    y_pred = (y_prob_oof >= threshold).astype(int)

    # Select cases
    if case_type == 'fn':
        mask = (y_oof == 1) & (y_pred == 0)
        title_prefix = "FALSE NEGATIVE (Missed Brugada)"
    elif case_type == 'fp':
        mask = (y_oof == 0) & (y_pred == 1)
        title_prefix = "FALSE POSITIVE (Misclassified Normal)"
    elif case_type == 'tp':
        mask = (y_oof == 1) & (y_pred == 1)
        title_prefix = "TRUE POSITIVE (Correct Brugada)"
    elif case_type == 'tn':
        mask = (y_oof == 0) & (y_pred == 0)
        title_prefix = "TRUE NEGATIVE (Correct Normal)"

    case_indices = np.where(mask)[0]
    if len(case_indices) == 0:
        print(f"No {case_type} cases found.")
        return

    # Select cases: sort FN/FP by confidence (most wrong = most interesting)
    if case_type == 'fn':
        # Lowest probability among Brugada = most missed
        sorted_cases = case_indices[np.argsort(y_prob_oof[case_indices])]
    elif case_type == 'fp':
        # Highest probability among Normal = most confidently wrong
        sorted_cases = case_indices[np.argsort(y_prob_oof[case_indices])[::-1]]
    else:
        sorted_cases = case_indices

    selected = sorted_cases[:n_cases]

    for rank, idx in enumerate(selected):
        pid = patient_ids[idx]
        true_label = 'Brugada' if y_oof[idx] == 1 else 'Normal'
        pred_prob = y_prob_oof[idx]

        # Build SHAP explanation object
        explanation = shap.Explanation(
            values=shap_vals[idx],
            base_values=np.mean(y_oof),   # Expected value approximation
            data=shap_result['X_oof'][idx],
            feature_names=feat_names
        )

        fig, ax = plt.subplots(figsize=(12, 8))
        shap.waterfall_plot(explanation, max_display=15, show=False)
        plt.title(
            f"{title_prefix}\n"
            f"Patient: {pid} | True: {true_label} | "
            f"P(Brugada)={pred_prob:.3f}",
            fontsize=11, pad=10
        )
        plt.tight_layout()
        save_path = f"{save_dir}/{case_type}_rank{rank+1}_pid{pid}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")


def _get_oof_probabilities(
    shap_result: Dict,
    feature_df: pd.DataFrame,
    model_factory,
    fold_df,
    feature_cols: List[str]
) -> np.ndarray:
    """
    Reconstruct OOF probabilities aligned with shap_result ordering.
    """
    from src.fold_manager import get_fold_split
    from src.feature_store import fit_scaler_imputer, transform

    patient_ids = shap_result['patient_ids']
    probs = np.zeros(len(patient_ids))
    n_folds = fold_df['fold_id'].nunique()

    for val_fold in range(n_folds):
        (X_train, y_train,
         X_val, y_val,
         train_ids, val_ids,
         feat_names) = get_fold_split(fold_df, feature_df, val_fold)

        feat_idx = [i for i, f in enumerate(feat_names) if f in feature_cols]
        X_train_sub = X_train[:, feat_idx]
        X_val_sub = X_val[:, feat_idx]

        imputer, scaler = fit_scaler_imputer(X_train_sub)
        X_train_proc = transform(X_train_sub, imputer, scaler)
        X_val_proc = transform(X_val_sub, imputer, scaler)

        model = model_factory()
        model.fit(X_train_proc, y_train)
        model_cal = calibrate_model(model, X_train_proc, y_train, 'sigmoid')
        fold_probs = model_cal.predict_proba(X_val_proc)[:, 1]

        for pid, prob in zip(val_ids, fold_probs):
            mask = patient_ids == pid
            probs[mask] = prob

    return probs