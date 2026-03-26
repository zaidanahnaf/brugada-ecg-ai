"""
tests/test_components.py

Unit tests for Brugada CNN components.
Run before any training to catch shape errors, leakage bugs, and contract violations.

Usage:
    python -m pytest tests/test_components.py -v
    python tests/test_components.py  (standalone)
"""

import os
import sys
import tempfile

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ===========================================================================
# PREPROCESSING TESTS
# ===========================================================================

class TestPreprocessing:

    def test_baseline_wander_shape_preserved(self):
        from src.cnn_preprocessing import remove_baseline_wander
        signals = np.random.randn(10, 3, 1200).astype(np.float32)
        result  = remove_baseline_wander(signals, kernel_samples=61)
        assert result.shape == signals.shape, f"Shape mismatch: {result.shape}"

    def test_baseline_wander_single_input(self):
        from src.cnn_preprocessing import remove_baseline_wander
        signal = np.random.randn(3, 1200).astype(np.float32)
        result = remove_baseline_wander(signal, kernel_samples=61)
        assert result.shape == signal.shape

    def test_lp_filter_shape(self):
        from src.cnn_preprocessing import apply_lowpass_filter
        signals = np.random.randn(5, 3, 1200).astype(np.float32)
        result  = apply_lowpass_filter(signals, cutoff_hz=40, order=4, fs=100)
        assert result.shape == signals.shape

    def test_normalizer_fit_on_train_only(self):
        """Verify normalizer.fit() uses train stats, and transform() doesn't refit."""
        from src.cnn_preprocessing import PerLeadNormalizer
        train = np.random.randn(20, 3, 1200).astype(np.float32) * 2.0 + 5.0
        val   = np.random.randn(5,  3, 1200).astype(np.float32) * 10.0

        norm = PerLeadNormalizer()
        norm.fit(train)
        train_proc = norm.transform(train)
        val_proc   = norm.transform(val)

        # After fitting on train, train should be ~zero mean, ~unit std
        assert abs(train_proc.mean()) < 0.1, "Train mean not near 0"
        assert abs(train_proc.std() - 1.0) < 0.1, "Train std not near 1"

        # Val is NOT normalized to zero mean/unit std — that's correct
        # (it uses train statistics, not val statistics)
        # But val_proc should have finite values
        assert np.isfinite(val_proc).all(), "Val has non-finite values"

    def test_normalizer_save_load(self):
        from src.cnn_preprocessing import PerLeadNormalizer
        signals = np.random.randn(10, 3, 1200).astype(np.float32)
        norm = PerLeadNormalizer()
        norm.fit(signals)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "norm_stats")
            norm.save(path)
            loaded = PerLeadNormalizer.load(path + ".npz")

        np.testing.assert_array_almost_equal(norm.mean_, loaded.mean_)
        np.testing.assert_array_almost_equal(norm.std_,  loaded.std_)

    def test_preprocess_fold_no_leakage(self):
        """Val transform must use train stats."""
        from src.cnn_preprocessing import preprocess_fold, PerLeadNormalizer
        # Train: all zeros + large constant (mean ≈ 5)
        train = np.ones((20, 3, 1200), dtype=np.float32) * 5.0
        val   = np.ones((5, 3, 1200), dtype=np.float32) * 100.0  # very different scale

        train_proc, val_proc, norm = preprocess_fold(train, val, apply_lp=False)

        # Train normalized to ~0 mean, unit std (after baseline removal, train is constant → std=0 edge)
        # Use random signals for a more meaningful test
        np.random.seed(42)
        train = np.random.randn(20, 3, 1200).astype(np.float32)
        val   = np.random.randn(5, 3, 1200).astype(np.float32) * 50.0 + 20.0

        train_proc, val_proc, norm = preprocess_fold(train, val, apply_lp=False)

        # Normalizer stats should come from train, not val
        for c in range(3):
            # If val had influenced stats, the huge mean (20) would be removed from val
            # We can verify stats match train
            train_flat = train[:, c, :].flatten()
            assert abs(norm.mean_[c] - train_flat.mean()) < 1.0  # within 1 mV


# ===========================================================================
# MODEL TESTS
# ===========================================================================

class TestModel:

    def test_forward_3lead(self):
        from src.cnn_model import build_model
        model = build_model(in_channels=3, embed_dim=64)
        model.eval()
        x = torch.randn(4, 3, 1200)
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (4, 1), f"Expected (4,1), got {logits.shape}"

    def test_forward_12lead(self):
        from src.cnn_model import build_model
        model = build_model(in_channels=12, embed_dim=64)
        model.eval()
        x = torch.randn(4, 12, 1200)
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (4, 1)

    def test_embedding_shape(self):
        from src.cnn_model import build_model
        model = build_model(in_channels=3, embed_dim=64)
        model.eval()
        x = torch.randn(8, 3, 1200)
        with torch.no_grad():
            emb = model.get_embedding(x)
        assert emb.shape == (8, 64), f"Expected (8,64), got {emb.shape}"

    def test_forward_with_embedding(self):
        from src.cnn_model import build_model
        model = build_model(in_channels=3, embed_dim=64)
        model.eval()
        x = torch.randn(4, 3, 1200)
        with torch.no_grad():
            logits, emb = model.forward_with_embedding(x)
        assert logits.shape == (4, 1)
        assert emb.shape == (4, 64)

    def test_embedding_consistency(self):
        """get_embedding and forward_with_embedding must return identical values."""
        from src.cnn_model import build_model
        model = build_model(in_channels=3, embed_dim=64)
        model.eval()
        x = torch.randn(4, 3, 1200)
        with torch.no_grad():
            emb_direct      = model.get_embedding(x)
            _, emb_combined = model.forward_with_embedding(x)
        torch.testing.assert_close(emb_direct, emb_combined)

    def test_single_sample_forward(self):
        """Edge case: batch size of 1."""
        from src.cnn_model import build_model
        model = build_model(in_channels=3, embed_dim=64)
        model.eval()
        x = torch.randn(1, 3, 1200)
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (1, 1)

    def test_parameter_count_reasonable(self):
        """Model should have < 500K parameters for 3-lead."""
        from src.cnn_model import build_model
        model = build_model(in_channels=3, embed_dim=64)
        n = model.count_parameters()
        assert n < 500_000, f"Model too large: {n} parameters"
        assert n > 50_000,  f"Model suspiciously small: {n} parameters"
        print(f"  3-lead model: {n:,} parameters")

    def test_loss_computes(self):
        """Verify BCEWithLogitsLoss with pos_weight works."""
        import torch.nn as nn
        from src.cnn_model import build_model
        model = build_model(in_channels=3)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.776]))
        x      = torch.randn(8, 3, 1200)
        labels = torch.randint(0, 2, (8, 1)).float()
        logits = model(x)
        loss   = criterion(logits, labels)
        assert loss.item() > 0
        assert torch.isfinite(loss)


# ===========================================================================
# DATASET TESTS
# ===========================================================================

class TestDataset:

    def test_dataset_basic(self):
        from src.cnn_dataset import BrugadaECGDataset
        signals = np.random.randn(10, 3, 1200).astype(np.float32)
        labels  = np.array([0]*7 + [1]*3, dtype=np.float32)
        ids     = [str(i) for i in range(10)]
        ds      = BrugadaECGDataset(signals, labels, ids, augment=False)

        assert len(ds) == 10
        sig, lbl = ds[0]
        assert sig.shape == (3, 1200)
        assert lbl.ndim == 0   # scalar

    def test_augmentation_changes_signal(self):
        from src.cnn_dataset import BrugadaECGDataset
        signals = np.ones((4, 3, 1200), dtype=np.float32)
        labels  = np.zeros(4, dtype=np.float32)
        ids     = [str(i) for i in range(4)]

        ds_aug   = BrugadaECGDataset(signals, labels, ids, augment=True, seed=42)
        ds_noaug = BrugadaECGDataset(signals, labels, ids, augment=False)

        sig_aug, _  = ds_aug[0]
        sig_noaug, _ = ds_noaug[0]
        # Augmented signal should differ from original
        assert not torch.allclose(sig_aug, sig_noaug)

    def test_patient_ids_are_strings(self):
        from src.cnn_dataset import BrugadaECGDataset
        signals = np.random.randn(5, 3, 1200).astype(np.float32)
        labels  = np.zeros(5, dtype=np.float32)
        ids     = [188981, 200000, 123456, 999999, 111111]  # integers
        ds      = BrugadaECGDataset(signals, labels, ids, augment=False)
        assert all(isinstance(p, str) for p in ds.patient_ids)

    def test_dataloader_batch_shape(self):
        from src.cnn_dataset import make_dataloader
        signals = np.random.randn(20, 3, 1200).astype(np.float32)
        labels  = np.zeros(20, dtype=np.float32)
        ids     = [str(i) for i in range(20)]
        loader  = make_dataloader(signals, labels, ids, batch_size=8, shuffle=False)

        batch_sigs, batch_labels = next(iter(loader))
        assert batch_sigs.shape == (8, 3, 1200)
        assert batch_labels.shape == (8,)


# ===========================================================================
# METRICS TESTS
# ===========================================================================

class TestMetrics:

    def test_perfect_predictions(self):
        from src.cnn_metrics import compute_metrics
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        m = compute_metrics(y_true, y_prob)
        assert m["auroc"] == 1.0
        assert m["sensitivity"] == 1.0
        assert m["specificity"] == 1.0

    def test_random_predictions(self):
        from src.cnn_metrics import compute_metrics
        np.random.seed(42)
        y_true = np.array([0]*50 + [1]*20)
        y_prob = np.random.rand(70)
        m = compute_metrics(y_true, y_prob)
        assert 0.0 <= m["auroc"] <= 1.0
        assert 0.0 <= m["sensitivity"] <= 1.0

    def test_aggregate(self):
        from src.cnn_metrics import aggregate_fold_metrics, compute_metrics
        np.random.seed(0)
        fold_metrics = []
        for _ in range(5):
            y_t = np.array([0]*50 + [1]*20)
            y_p = np.random.rand(70)
            fold_metrics.append(compute_metrics(y_t, y_p))
        summary = aggregate_fold_metrics(fold_metrics)
        assert "auroc_mean" in summary
        assert "auroc_std" in summary


# ===========================================================================
# UTILS TESTS
# ===========================================================================

class TestUtils:

    def test_sha256_wrong_raises(self):
        from src.cnn_utils import verify_fold_sha256
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("patient_id,fold_id\n1,0\n")
            path = f.name
        try:
            with pytest.raises(RuntimeError):
                verify_fold_sha256(path, "wrong_sha256_value")
        finally:
            os.unlink(path)

    def test_set_seed_reproducible(self):
        from src.cnn_utils import set_seed
        set_seed(42)
        a = np.random.randn(5)
        set_seed(42)
        b = np.random.randn(5)
        np.testing.assert_array_equal(a, b)


# ===========================================================================
# STANDALONE RUNNER
# ===========================================================================

if __name__ == "__main__":
    import traceback
    test_classes = [
        TestPreprocessing, TestModel, TestDataset, TestMetrics, TestUtils
    ]
    passed = failed = 0
    for cls in test_classes:
        instance = cls()
        for name in [m for m in dir(instance) if m.startswith("test_")]:
            try:
                getattr(instance, name)()
                print(f"  PASS  {cls.__name__}.{name}")
                passed += 1
            except Exception as e:
                print(f"  FAIL  {cls.__name__}.{name}: {e}")
                traceback.print_exc()
                failed += 1

    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed > 0:
        sys.exit(1)