"""Reads config.yaml and hands each component its own typed config."""
from dataclasses import dataclass, field
from pathlib import Path

from src.constant import CONFIG_FILE_PATH, ROOT_DIR
from src.utils.common import read_yaml


@dataclass(frozen=True)
class DataIngestionConfig:
    raw_dir: Path
    processed_dir: Path
    user_sample_mod: int
    min_user_interactions: int
    min_item_interactions: int
    chunk_size: int


@dataclass(frozen=True)
class DataTransformationConfig:
    processed_dir: Path
    artifacts_dir: Path
    val_per_user: int
    test_per_user: int


@dataclass(frozen=True)
class ModelTrainerConfig:
    artifacts_dir: Path
    k: int
    selection_metric: str
    tuning_sample_users: int
    batch_size: int
    random_state: int
    models: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ModelEvaluationConfig:
    artifacts_dir: Path
    k: int
    batch_size: int
    selection_metric: str


class ConfigurationManager:
    def __init__(self, config_path: Path = CONFIG_FILE_PATH):
        self.config = read_yaml(config_path)

    @staticmethod
    def _path(value: str) -> Path:
        return ROOT_DIR / value

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        cfg = self.config["data_ingestion"]
        return DataIngestionConfig(
            raw_dir=self._path(cfg["raw_dir"]),
            processed_dir=self._path(cfg["processed_dir"]),
            user_sample_mod=cfg["user_sample_mod"],
            min_user_interactions=cfg["min_user_interactions"],
            min_item_interactions=cfg["min_item_interactions"],
            chunk_size=cfg["chunk_size"],
        )

    def get_data_transformation_config(self) -> DataTransformationConfig:
        cfg = self.config["data_transformation"]
        return DataTransformationConfig(
            processed_dir=self._path(cfg["processed_dir"]),
            artifacts_dir=self._path(cfg["artifacts_dir"]),
            val_per_user=cfg["val_per_user"],
            test_per_user=cfg["test_per_user"],
        )

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        cfg = self.config["model_trainer"]
        return ModelTrainerConfig(
            artifacts_dir=self._path(cfg["artifacts_dir"]),
            k=cfg["k"],
            selection_metric=cfg["selection_metric"],
            tuning_sample_users=cfg["tuning_sample_users"],
            batch_size=cfg["batch_size"],
            random_state=cfg["random_state"],
            models=cfg["models"],
        )

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        cfg = self.config["model_evaluation"]
        return ModelEvaluationConfig(
            artifacts_dir=self._path(cfg["artifacts_dir"]),
            k=cfg["k"],
            batch_size=cfg["batch_size"],
            selection_metric=cfg["selection_metric"],
        )
