# tests/test_leakage_guard.py — add this test case

def test_undersample_strips_class_weight():
    """
    Confirm that _strip_class_weight removes class_weight
    without mutating the original config.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.base import clone
    from experiments.models.classical_ml import _strip_class_weight

    original_config = {
        'name': 'TestLR',
        'estimator': LogisticRegression(class_weight='balanced'),
        'param_grid': {'C': [0.1]},
        'search_type': 'grid'
    }

    stripped_config = _strip_class_weight(original_config)

    # Original must be unchanged
    assert original_config['estimator'].class_weight == 'balanced', \
        "Original config was mutated — clone() not working"

    # Stripped must have class_weight=None
    assert stripped_config['estimator'].class_weight is None, \
        "class_weight not stripped from cloned estimator"

    # Config name preserved
    assert stripped_config['name'] == 'TestLR'

    print("test_undersample_strips_class_weight: PASSED")


def test_undersample_strips_xgb_scale_pos_weight():
    try:
        from xgboost import XGBClassifier
        from experiments.models.classical_ml import _strip_class_weight

        original_config = {
            'name': 'TestXGB',
            'estimator': XGBClassifier(scale_pos_weight=3.77),
            'param_grid': {},
            'search_type': 'random',
            'n_iter': 5
        }
        stripped = _strip_class_weight(original_config)
        assert stripped['estimator'].scale_pos_weight == 1, \
            "scale_pos_weight not reset to 1"
        assert original_config['estimator'].scale_pos_weight == 3.77, \
            "Original XGB config mutated"
        print("test_undersample_strips_xgb_scale_pos_weight: PASSED")
    except ImportError:
        print("XGBoost not installed — test skipped")