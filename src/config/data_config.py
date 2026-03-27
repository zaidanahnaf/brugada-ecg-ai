from dataclasses import dataclass, field
from typing import List

@dataclass
class DataConfig:
    data_dir: str = "data/raw/brugada/files"
    metadata_path: str = "data/raw/brugada/metadata.csv"
    fs: int = 100
    signal_duration_s: float = 12.0
    n_leads: int = 12
    lead_names: List[str] = field(default_factory=lambda: [
        'I','II','III','aVR','aVL','aVF',
        'V1','V2','V3','V4','V5','V6'
    ])
    priority_leads: List[str] = field(default_factory=lambda: ['V1','V2','V3'])
    target_col: str = 'brugada'