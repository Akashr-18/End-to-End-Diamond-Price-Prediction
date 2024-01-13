from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir : Path
    source_url : str
    local_data_file : Path
    unzip_dir : Path
    train_data_file_path: Path
    test_data_file_path: Path

@dataclass(frozen=True)
class DataPreprocessingConfig:
    root_dir : Path
    
@dataclass(frozen=True)
class ModelTrainingConfig:
    root_dir : Path
    plot_dir : Path
    plot_file_path: Path

@dataclass(frozen=True)
class ModelParametersConfig:
    xgboost_n_estimators : list
    xgboost_learning_rate: list
    xgboost_max_depth: list
    xgboost_min_child_weight: list
    xgboost_gamma: list
    xgboost_colsample_bytree: list
    xgboost_subsample: list
    xgboost_reg_alpha: list
    xgboost_reg_lambda: list
    # randomforest_n_estimators: list
    # randomforest_criterion: list
    # randomforest_max_depth: list
    # randomforest_min_samples_split: list
    # randomforest_min_samples_leaf: list
    # randomforest_bootstrap: list