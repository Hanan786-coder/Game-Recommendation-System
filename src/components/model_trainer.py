"""Tunes every model on the validation split, then refits them for testing and serving."""
from itertools import product

import numpy as np
import scipy.sparse as sp

from src.components.data_transformation import InteractionData
from src.components.model_evaluation import evaluate_model, users_with_targets
from src.components.recommenders import (BaseRecommender, ContentBasedRecommender, ItemKNNRecommender,
                                         build_model)
from src.configuration.configuration import ModelTrainerConfig
from src.constant import HISTORY_FILE, MODEL_FILE, SEEN_FILE, SIMILARITY_MODEL_FILE, TUNING_REPORT_FILE
from src.exception import CustomException
from src.logger import logger
from src.utils.common import save_json, save_object


def param_grid(spec: dict) -> list[dict]:
    """Expands {'params': defaults, 'grid': {name: [values]}} into a list of parameter sets."""
    base, grid = spec.get("params") or {}, spec.get("grid") or {}
    if not grid:
        return [dict(base)]
    return [{**base, **dict(zip(grid, combo))} for combo in product(*grid.values())]


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def tune(self, data: InteractionData) -> tuple[dict[str, dict], str]:
        """Grid search per model on train -> validation.

        Returns the best params per model and the name of the model with the best validation score.
        """
        try:
            history, seen = data.matrices(data.train)
            target, _ = data.matrices(data.val)

            # A fixed sample of validation users keeps the search fast and comparable
            users = users_with_targets(target)
            rng = np.random.default_rng(self.config.random_state)
            if len(users) > self.config.tuning_sample_users:
                users = np.sort(rng.choice(users, self.config.tuning_sample_users, replace=False))
            logger.info(f"Tuning on {len(users):,} validation users")

            metric, results, best_params = self.config.selection_metric, {}, {}
            for name, spec in self.config.models.items():
                trials = []
                for params in param_grid(spec):
                    model = build_model(name, **params).fit(history, data.items)
                    metrics = evaluate_model(model, history, seen, target, users, self.config.k,
                                             self.config.batch_size)
                    trials.append({"params": params, "metrics": metrics})
                    logger.info(f"[val] {name} {params}: {metric}@{self.config.k}={metrics[metric]}")

                best = max(trials, key=lambda t: t["metrics"][metric])
                best_params[name] = best["params"]
                results[name] = {"best_params": best["params"], "best_metrics": best["metrics"], "trials": trials}

            # Model selection happens here, on validation, so the test split stays untouched
            selected = max(results, key=lambda name: results[name]["best_metrics"][metric])
            logger.info(f"Selected model on validation: {selected}")

            save_json({"k": self.config.k, "selection_metric": metric, "n_users": int(len(users)),
                       "selected_model": selected, "models": results},
                      self.config.artifacts_dir / TUNING_REPORT_FILE)
            return best_params, selected
        except Exception as e:
            raise CustomException(e) from e

    def fit_models(self, best_params: dict[str, dict], interactions: sp.csr_matrix,
                   data: InteractionData) -> dict[str, BaseRecommender]:
        """Refits each model with its tuned params on the given interactions."""
        models = {}
        for name, params in best_params.items():
            logger.info(f"Fitting {name} with {params}")
            models[name] = build_model(name, **params).fit(interactions, data.items)
        return models

    def train_serving_model(self, name: str, best_params: dict[str, dict], data: InteractionData) -> None:
        """Retrains the selected model on all interactions and saves everything serving needs."""
        try:
            history, seen = data.matrices(data.train, data.val, data.test)
            artifacts_dir = self.config.artifacts_dir

            model = build_model(name, **best_params[name]).fit(history, data.items)
            save_object(model, artifacts_dir / MODEL_FILE)

            # "Games like this one": content similarity first, co-play similarity for games without metadata
            similarity_models = {
                "content": ContentBasedRecommender(**best_params.get(ContentBasedRecommender.name, {})),
                "collaborative": ItemKNNRecommender(**best_params.get(ItemKNNRecommender.name, {})),
            }
            for similarity_model in similarity_models.values():
                similarity_model.fit(history, data.items)
            save_object(similarity_models, artifacts_dir / SIMILARITY_MODEL_FILE)

            sp.save_npz(artifacts_dir / HISTORY_FILE, history)
            sp.save_npz(artifacts_dir / SEEN_FILE, seen)
            logger.info(f"Saved serving model ({model}) to {artifacts_dir / MODEL_FILE}")
        except Exception as e:
            raise CustomException(e) from e
