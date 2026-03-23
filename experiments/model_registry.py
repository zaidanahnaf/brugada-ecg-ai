# experiments/model_registry.py

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import numpy as np

# Optional — check availability at runtime
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except ImportError:
    HAS_CAT = False

try:
    from imblearn.ensemble import BalancedRandomForestClassifier
    HAS_BRF = True
except ImportError:
    HAS_BRF = False


# ── Model 1: Logistic Regression ─────────────────────────────────
# LR_BALANCED = {
#     'name': 'LogReg_Balanced',
#     'estimator': LogisticRegression(
#         class_weight='balanced',
#         max_iter=5000,
#         solver='saga',
#         random_state=42
#     ),
#     'param_grid': {
#         'C': [0.001, 0.01, 0.1, 1.0, 10.0],
#         'l1_ratio': ['0.0', '1.0'],
#     },
#     'search_type': 'grid',
#     # 'notes': 'L1 = feature selection; L2 = shrinkage. Both evaluated.'
#     'notes': 'l1_ratio=1.0 → L1 sparsity; l1_ratio=0.0 → L2 shrinkage'
# }

# LR_UNWEIGHTED = {
#     'name': 'LogReg_Unweighted',
#     'estimator': LogisticRegression(
#         max_iter=5000,
#         solver='saga',
#         random_state=42
#     ),
#     'param_grid': {
#         'C': [0.001, 0.01, 0.1, 1.0, 10.0],
#         'l1_ratio': ['0.0', '1.0'],
#     },
#     'search_type': 'grid',
#     'notes': 'Ablation: does class_weight matter for this dataset?'
# }

LR_BALANCED = {
    'name': 'LogReg_Balanced_L1',
    'estimator': LogisticRegression(
        class_weight='balanced',
        max_iter=5000,
        solver='saga',
        l1_ratio=1.0,          # L1 regularization
        random_state=42
    ),
    'param_grid': {
        'C': [0.001, 0.01, 0.1, 1.0, 10.0],
    },
    'search_type': 'grid',
    'notes': 'L1 regularization via l1_ratio=1.0 — feature selection effect'
}

LR_BALANCED_L2 = {
    'name': 'LogReg_Balanced_L2',
    'estimator': LogisticRegression(
        class_weight='balanced',
        max_iter=5000,
        solver='saga',
        l1_ratio=0.0,          # L2 regularization
        random_state=42
    ),
    'param_grid': {
        'C': [0.001, 0.01, 0.1, 1.0, 10.0],
    },
    'search_type': 'grid',
    'notes': 'L2 regularization via l1_ratio=0.0 — shrinkage'
}

LR_UNWEIGHTED = {
    'name': 'LogReg_Unweighted_L1',
    'estimator': LogisticRegression(
        max_iter=5000,
        solver='saga',
        l1_ratio=1.0,
        random_state=42
    ),
    'param_grid': {
        'C': [0.001, 0.01, 0.1, 1.0, 10.0],
    },
    'search_type': 'grid',
    'notes': 'Ablation: no class weight'
}

# ── Model 2: Linear SVM ───────────────────────────────────────────
SVM_LINEAR = {
    'name': 'SVM_Linear_Balanced',
    'estimator': SVC(
        kernel='linear',
        class_weight='balanced',
        probability=True,    # Required for AUROC / calibration
        random_state=42
    ),
    'param_grid': {
        'C': [0.001, 0.01, 0.1, 1.0, 10.0],
    },
    'search_type': 'grid',
    'notes': 'Linear boundary — interpretable coefficient weights'
}

# ── Model 3: RBF SVM ──────────────────────────────────────────────
SVM_RBF = {
    'name': 'SVM_RBF_Balanced',
    'estimator': SVC(
        kernel='rbf',
        class_weight='balanced',
        probability=True,
        random_state=42
    ),
    'param_grid': {
        'C': [0.1, 1.0, 10.0, 100.0],
        'gamma': ['scale', 'auto', 0.001, 0.01],
    },
    'search_type': 'grid',
    'notes': 'Non-linear; risk of overfit on small dataset'
}

# ── Model 4: Random Forest ────────────────────────────────────────
RF_BALANCED = {
    'name': 'RF_Balanced',
    'estimator': RandomForestClassifier(
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1
    ),
    'param_grid': {
        'n_estimators': [100, 300, 500],
        'max_depth': [3, 5, 8, None],
        'min_samples_leaf': [1, 3, 5],
        'max_features': ['sqrt', 'log2'],
    },
    'search_type': 'random',
    'n_iter': 30,
    'notes': (
        'balanced_subsample reweights each bootstrap independently. '
        'Constrain depth to reduce overfit.'
    )
}

# ── Model 5: XGBoost ─────────────────────────────────────────────
XGB_BALANCED = {
    'name': 'XGB_Balanced',
    'estimator': XGBClassifier(
        scale_pos_weight=287/76,   # n_negative / n_positive (approx)
        eval_metric='aucpr',
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1
    ) if HAS_XGB else None,
    'param_grid': {
        'n_estimators': [100, 200, 400],
        'max_depth': [2, 3, 4, 6],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'min_child_weight': [1, 3, 5],
        'reg_alpha': [0, 0.1, 1.0],        # L1
        'reg_lambda': [1.0, 5.0, 10.0],   # L2
    },
    'search_type': 'random',
    'n_iter': 40,
    'notes': (
        'scale_pos_weight handles imbalance for XGB. '
        'Heavy regularization grid needed for small dataset.'
    )
}

# ── Model 6: LightGBM ─────────────────────────────────────────────
LGB_BALANCED = {
    'name': 'LGB_Balanced',
    'estimator': LGBMClassifier(
        is_unbalance=True,
        metric='average_precision',
        random_state=42,
        n_jobs=-1,
        verbose=-1
    ) if HAS_LGB else None,
    'param_grid': {
        'n_estimators': [100, 200, 400],
        'max_depth': [3, 4, 6, -1],
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [15, 31, 63],
        'min_child_samples': [5, 10, 20],
        'reg_alpha': [0, 0.1, 1.0],
        'reg_lambda': [0, 1.0, 5.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
    },
    'search_type': 'random',
    'n_iter': 40,
    'notes': 'Faster than XGB. is_unbalance=True rescales loss by class freq.'
}

# ── Model 7: Balanced Random Forest ──────────────────────────────
BRF = {
    'name': 'BalancedRF',
    'estimator': BalancedRandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        sampling_strategy='auto',
        replacement=False
    ) if HAS_BRF else None,
    'param_grid': {
        'n_estimators': [100, 300, 500],
        'max_depth': [3, 5, 8, None],
        'min_samples_leaf': [1, 3, 5],
        'max_features': ['sqrt', 'log2'],
    },
    'search_type': 'random',
    'n_iter': 20,
    'notes': (
        'Undersamples majority class per bootstrap. '
        'Undersampling is internal to each tree — not pre-split. '
        'Compare vs RF_Balanced to assess undersampling effect.'
    )
}

# ── Model 8: CatBoost ────────────────────────────────────────────
CATBOOST = {
    'name': 'CatBoost_Balanced',
    'estimator': CatBoostClassifier(
        auto_class_weights='Balanced',
        eval_metric='AUC',
        random_seed=42,
        verbose=0
    ) if HAS_CAT else None,
    'param_grid': {
        'iterations': [100, 200, 400],
        'depth': [3, 4, 6],
        'learning_rate': [0.01, 0.05, 0.1],
        'l2_leaf_reg': [1, 3, 5, 10],
        'border_count': [32, 64, 128],
    },
    'search_type': 'random',
    'n_iter': 30,
    'notes': 'Good with small tabular datasets. Symmetric trees reduce overfit.'
}


def get_all_models() -> list:
    """Return list of valid model configs (skip None estimators)."""
    all_configs = [
        LR_BALANCED, LR_BALANCED_L2, LR_UNWEIGHTED,
        SVM_LINEAR, SVM_RBF,
        RF_BALANCED, BRF,
        XGB_BALANCED, LGB_BALANCED, CATBOOST
    ]
    return [m for m in all_configs if m.get('estimator') is not None]