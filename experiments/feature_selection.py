# experiments/feature_selection.py

import numpy as np
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif,
    SelectFromModel, RFE
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from typing import Tuple, List, Optional


def select_features_inside_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    feature_names: List[str],
    method: str = 'univariate_f',
    k: int = 40
) -> Tuple[np.ndarray, np.ndarray, List[str], object]:
    """
    Feature selection — fit on X_train ONLY, transform both splits.

    Methods
    -------
    'univariate_f'     : ANOVA F-score SelectKBest
    'mutual_info'      : Mutual information SelectKBest
    'l1_lr'            : L1 logistic regression SelectFromModel
    'rf_importance'    : Random Forest importance threshold
    'none'             : No selection (return all features)

    Returns
    -------
    X_train_sel, X_val_sel, selected_feature_names, selector_object
    """
    if method == 'none':
        return X_train, X_val, feature_names, None

    elif method == 'univariate_f':
        k_safe = min(k, X_train.shape[1])
        selector = SelectKBest(f_classif, k=k_safe)
        X_train_sel = selector.fit_transform(X_train, y_train)
        X_val_sel = selector.transform(X_val)
        selected = [feature_names[i] for i in selector.get_support(indices=True)]

    elif method == 'mutual_info':
        k_safe = min(k, X_train.shape[1])
        selector = SelectKBest(mutual_info_classif, k=k_safe)
        X_train_sel = selector.fit_transform(X_train, y_train)
        X_val_sel = selector.transform(X_val)
        selected = [feature_names[i] for i in selector.get_support(indices=True)]

    elif method == 'l1_lr':
        base = LogisticRegression(
            penalty='l1', solver='saga', C=0.1,
            class_weight='balanced', max_iter=2000, random_state=42
        )
        selector = SelectFromModel(base, threshold='mean')
        X_train_sel = selector.fit_transform(X_train, y_train)
        X_val_sel = selector.transform(X_val)
        selected = [feature_names[i] for i in selector.get_support(indices=True)]

    elif method == 'rf_importance':
        base = RandomForestClassifier(
            n_estimators=200, class_weight='balanced',
            random_state=42, n_jobs=-1
        )
        selector = SelectFromModel(base, threshold='mean')
        X_train_sel = selector.fit_transform(X_train, y_train)
        X_val_sel = selector.transform(X_val)
        selected = [feature_names[i] for i in selector.get_support(indices=True)]

    else:
        raise ValueError(f"Unknown selection method: {method}")

    print(f"  Feature selection [{method}]: {X_train.shape[1]} → {X_train_sel.shape[1]}")
    return X_train_sel, X_val_sel, selected, selector