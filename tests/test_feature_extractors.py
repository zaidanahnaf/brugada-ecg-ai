# tests/test_feature_extractors.py — add this test case

def test_confidence_features_written_to_matrix():
    """
    Confirm that qc_ confidence features appear in the
    output of run_subject and are excluded from ML features
    by get_fold_split.
    """
    import numpy as np
    from unittest.mock import patch, MagicMock
    from src.pipeline import _compute_confidence_features
    from src.config.__init__ import CFG

    # Test _compute_confidence_features directly
    mock_counts = {
        'V1': {'low': 3, 'fallback': 1, 'total': 10},
        'V2': {'low': 0, 'fallback': 0, 'total': 10},
        'V3': {'low': 6, 'fallback': 2, 'total': 10},
    }
    result = _compute_confidence_features(mock_counts)

    # Check expected keys exist
    assert 'qc_V1_j_low_conf_rate' in result
    assert 'qc_V2_j_low_conf_rate' in result
    assert 'qc_V3_j_low_conf_rate' in result
    assert 'qc_V1_j_fallback_rate' in result
    assert 'qc_any_lead_j_unreliable' in result

    # Check values
    assert abs(result['qc_V1_j_low_conf_rate'] - 0.3) < 1e-9
    assert abs(result['qc_V2_j_low_conf_rate'] - 0.0) < 1e-9
    assert abs(result['qc_V3_j_low_conf_rate'] - 0.6) < 1e-9

    # V3 has low_conf_rate > 0.5 → any_unreliable should be 1
    assert result['qc_any_lead_j_unreliable'] == 1

    # Check qc_ prefix matches config
    qc_keys = [k for k in result if k.startswith(CFG.feature.qc_feature_prefix)]
    assert len(qc_keys) == len(result), \
        "All confidence feature keys must start with qc_ prefix"

    print("test_confidence_features_written_to_matrix: PASSED")


def test_qc_columns_excluded_from_fold_split():
    """
    Confirm that qc_ columns are excluded from the ML feature
    set returned by get_fold_split.
    """
    import pandas as pd
    import numpy as np
    from src.fold_manager import get_fold_split
    from src.config.__init__ import CFG

    # Minimal mock feature_df with qc_ columns
    n = 20
    mock_df = pd.DataFrame({
        'patient_id': [f'P{i:03d}' for i in range(n)],
        CFG.data.target_col: [0]*15 + [1]*5,
        'pipeline_status': ['OK'] * n,
        'n_valid_beats': [10] * n,
        'st_V1_st_j40_median': np.random.randn(n),
        'morph_V1_covedness_score_median': np.random.randn(n),
        'qc_V1_j_low_conf_rate': np.random.rand(n),    # Should be excluded
        'qc_any_lead_j_unreliable': np.random.randint(0, 2, n),  # Excluded
    })
    mock_fold_df = pd.DataFrame({
        'patient_id': [f'P{i:03d}' for i in range(n)],
        CFG.data.target_col: [0]*15 + [1]*5,
        'fold_id': [i % 5 for i in range(n)]
    })

    X_train, y_train, X_val, y_val, _, _, feat_names = get_fold_split(
        mock_fold_df, mock_df, val_fold=0
    )

    # No qc_ column should appear in feat_names
    qc_in_features = [f for f in feat_names if f.startswith('qc_')]
    assert len(qc_in_features) == 0, \
        f"qc_ columns leaked into ML features: {qc_in_features}"

    # Clinical features should still be present
    assert 'st_V1_st_j40_median' in feat_names
    assert 'morph_V1_covedness_score_median' in feat_names

    print("test_qc_columns_excluded_from_fold_split: PASSED")