"""
src/utils.py
Utility functions: reproducibility, SHA256 verification, logging, directory helpers.
"""

import hashlib
import json
import logging
import os
import random
import sys

import numpy as np


# =============================================================================
# LOGGING
# =============================================================================

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("[%(asctime)s %(levelname)s %(name)s] %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_seed(seed: int) -> None:
    """Fix all random seeds for reproducibility. Call before any fold."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# =============================================================================
# FILE SYSTEM
# =============================================================================

def ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


# =============================================================================
# SHA256 VERIFICATION
# =============================================================================

def sha256_file(filepath: str) -> str:
    """Return hex SHA256 digest of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_fold_sha256(filepath: str, expected: str, logger=None) -> bool:
    """
    Verify fold_assignments.csv SHA256.
    Raises RuntimeError on mismatch — this is a hard stop, not a warning.
    """
    log = logger or get_logger("verify_sha256")
    actual = sha256_file(filepath)
    if actual != expected:
        msg = (
            f"SHA256 MISMATCH on {filepath}\n"
            f"  Expected : {expected}\n"
            f"  Actual   : {actual}\n"
            "TRAINING ABORTED — fold file may have been tampered with or regenerated."
        )
        log.error(msg)
        raise RuntimeError(msg)
    log.info(f"SHA256 OK: {filepath}")
    return True


# =============================================================================
# JSON HELPERS
# =============================================================================

def save_json(obj: dict, path: str) -> None:
    ensure_dirs(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)