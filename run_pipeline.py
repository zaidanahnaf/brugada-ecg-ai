# run_pipeline.py

import logging
import pandas as pd
from src.config.__init__ import CFG
from src.data.data_loader import load_metadata
from src.fold_manager import create_folds
from src.pipeline import run_full_pipeline

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

if __name__ == "__main__":
    main()