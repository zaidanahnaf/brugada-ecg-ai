# run_pipeline.py

import logging
from pathlib import Path
import pandas as pd
from src.config.__init__ import CFG
from src.data.data_loader import load_metadata
from src.fold_manager import create_folds
from src.pipeline import run_full_pipeline
from src.feature_store import (
        get_clean_feature_matrix,
        export_scaled_feature_matrices
    )

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s — %(message)s',
    handlers=[
        logging.FileHandler(f"{CFG.feature.logs_dir}/pipeline.log"),
        logging.StreamHandler()
    ]
)

def main():
    # Load metadata
    metadata = load_metadata(CFG.data.metadata_path)
    patient_ids = metadata['patient_id'].astype(str).tolist()

    # Create folds ONCE — deterministic
    fold_df = create_folds(metadata)

    # Run extraction
    feature_df = run_full_pipeline(
        patient_ids=patient_ids,
        metadata=metadata,
        data_dir=CFG.data.data_dir,
        output_dir=CFG.feature.features_dir
    )

    print(f"\nFeature matrix shape: {feature_df.shape}")
    print(f"FAILED subjects: {(feature_df['pipeline_status']=='FAILED').sum()}")
    print(f"PARTIAL subjects: {(feature_df['pipeline_status']=='PARTIAL').sum()}")
    print(f"Feature columns: {feature_df.shape[1] - 4}")  # minus meta cols

    # Feature store   
    exclude_cols = [
        'patient_id',
        CFG.data.target_col,
        'pipeline_status',
        'n_valid_beats'
    ]

    feature_cols = [c for c in feature_df.columns if c not in exclude_cols]

    if len(feature_cols) == 0:
        raise ValueError("No feature columns found.")

    feature_df, feature_cols = get_clean_feature_matrix(
        feature_df,
        feature_cols
    )

    if len(feature_cols) == 0:
        raise ValueError("All features dropped after cleaning.")

    export_scaled_feature_matrices(
        feature_df=feature_df,
        fold_df=fold_df,
        feature_cols=feature_cols,
        output_dir="outputs/features/scaled"
    )

    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_path = processed_dir / "feature_matrix.csv"

    feature_df.to_csv(processed_path, index=False)
    print(f"[OK] feature_matrix saved to {processed_path}")

if __name__ == "__main__":
    main()