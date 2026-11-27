"""End-to-end training: ingest -> transform -> tune -> evaluate -> save serving model.

Usage:
    python -m src.pipeline.training_pipeline
    python -m src.pipeline.training_pipeline --force-ingestion   # rebuild processed data
"""
import argparse
import time

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_evaluation import ModelEvaluation
from src.components.model_trainer import ModelTrainer
from src.configuration.configuration import ConfigurationManager
from src.logger import logger


class TrainingPipeline:
    def __init__(self, config_manager: ConfigurationManager | None = None):
        self.config = config_manager or ConfigurationManager()

    def run(self, force_ingestion: bool = False) -> dict:
        start = time.perf_counter()

        logger.info("Stage 1/4: data ingestion")
        DataIngestion(self.config.get_data_ingestion_config()).initiate_data_ingestion(force=force_ingestion)

        logger.info("Stage 2/4: data transformation")
        data = DataTransformation(self.config.get_data_transformation_config()).initiate_data_transformation()

        logger.info("Stage 3/4: hyperparameter tuning on validation")
        trainer = ModelTrainer(self.config.get_model_trainer_config())
        best_params, selected = trainer.tune(data)

        logger.info("Stage 4/4: evaluation on test")
        history, _ = data.matrices(data.train, data.val)
        models = trainer.fit_models(best_params, history, data)
        report = ModelEvaluation(self.config.get_model_evaluation_config()).evaluate_on_test(
            models, data, selected)

        # The selected model is retrained on every interaction before it is saved for serving
        trainer.train_serving_model(selected, best_params, data)
        logger.info(f"Training pipeline finished in {(time.perf_counter() - start) / 60:.1f} min")
        return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the game recommender")
    parser.add_argument("--force-ingestion", action="store_true", help="re-read the raw CSV files")
    args = parser.parse_args()
    TrainingPipeline().run(force_ingestion=args.force_ingestion)
