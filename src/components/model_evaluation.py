"""Offline evaluation: rank unseen games for each user and compare with their held-out games."""
from collections import defaultdict

import numpy as np
import scipy.sparse as sp

from src.components.data_transformation import InteractionData
from src.components.recommenders import BaseRecommender
from src.configuration.configuration import ModelEvaluationConfig
from src.constant import EVALUATION_REPORT_FILE
from src.exception import CustomException
from src.logger import logger
from src.utils.common import save_json
from src.utils.metrics import hit_matrix, ranking_metrics


def users_with_targets(target: sp.csr_matrix) -> np.ndarray:
    """Users that have at least one liked game in the held-out split."""
    return np.flatnonzero(target.getnnz(axis=1) > 0)


def evaluate_model(model: BaseRecommender, history: sp.csr_matrix, seen: sp.csr_matrix,
                   target: sp.csr_matrix, users: np.ndarray, k: int = 10, batch_size: int = 2048) -> dict:
    """Mean ranking metrics @K over `users`, plus catalogue coverage."""
    totals = defaultdict(float)
    recommended = np.zeros(target.shape[1], dtype=bool)

    for start in range(0, len(users), batch_size):
        batch = users[start:start + batch_size]
        recs, _ = model.recommend(history[batch], k=k, exclude=seen[batch], user_idx=batch)
        relevant = target[batch].toarray() > 0

        for name, values in ranking_metrics(hit_matrix(recs, relevant), relevant.sum(axis=1)).items():
            totals[name] += float(values.sum())
        recommended[recs.ravel()] = True

    metrics = {name: round(total / len(users), 5) for name, total in totals.items()}
    metrics["coverage"] = round(float(recommended.mean()), 5)  # share of the catalogue ever recommended
    return metrics


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def evaluate_on_test(self, models: dict[str, BaseRecommender], data: InteractionData) -> dict:
        """Scores models fitted on train+val against each user's most recent (test) games."""
        try:
            history, seen = data.matrices(data.train, data.val)
            target, _ = data.matrices(data.test)
            users = users_with_targets(target)
            logger.info(f"Evaluating {len(models)} models on {len(users):,} test users (K={self.config.k})")

            results = {}
            for name, model in models.items():
                metrics = evaluate_model(model, history, seen, target, users, self.config.k,
                                         self.config.batch_size)
                results[name] = {"params": model.get_params(), "metrics": metrics}
                logger.info(f"[test] {name}: {metrics}")

            metric = self.config.selection_metric
            best = max(results, key=lambda name: results[name]["metrics"][metric])
            report = {"k": self.config.k, "split": "test", "n_users": int(len(users)),
                      "selection_metric": metric, "best_model": best, "models": results}

            save_json(report, self.config.artifacts_dir / EVALUATION_REPORT_FILE)
            logger.info(f"Best model on test: {best} ({metric}@{self.config.k} = "
                        f"{results[best]['metrics'][metric]})")
            return report
        except Exception as e:
            raise CustomException(e) from e
