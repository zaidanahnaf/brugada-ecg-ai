from .data_config import DataConfig
from .preprocessing_config import PreprocessingConfig
from .feature_config import FeatureConfig

class PipelineConfig:
    def __init__(self):
        self.data = DataConfig()
        self.preprocessing = PreprocessingConfig()
        self.feature = FeatureConfig()

CFG = PipelineConfig()