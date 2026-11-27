"""Loads the trained artifacts and serves recommendations.

Usage:
    python -m src.pipeline.prediction_pipeline --user-id 51580
    python -m src.pipeline.prediction_pipeline --liked 1145360 367520     # new user
    python -m src.pipeline.prediction_pipeline --similar-to 1145360
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.constant import (HISTORY_FILE, ITEM_COL, ITEMS_FILE, MAPPINGS_FILE, MODEL_FILE, ROOT_DIR, SEEN_FILE,
                          SIMILARITY_MODEL_FILE)
from src.exception import CustomException
from src.utils.common import load_object

OUTPUT_COLUMNS = ["rank", ITEM_COL, "title", "score", "rating", "price_final"]


class PredictionPipeline:
    def __init__(self, artifacts_dir: Path = ROOT_DIR / "artifacts"):
        try:
            self.model = load_object(artifacts_dir / MODEL_FILE)
            self.similarity_model = load_object(artifacts_dir / SIMILARITY_MODEL_FILE)
            mappings = load_object(artifacts_dir / MAPPINGS_FILE)
            self.items = pd.read_parquet(artifacts_dir / ITEMS_FILE)
            self.history = sp.load_npz(artifacts_dir / HISTORY_FILE).tocsr()
            self.seen = sp.load_npz(artifacts_dir / SEEN_FILE).tocsr()
        except Exception as e:
            raise CustomException(e) from e

        # Lookup tables between raw ids and matrix indices
        self.user_index = {uid: idx for idx, uid in enumerate(mappings["user_ids"])}
        self.item_index = {iid: idx for idx, iid in enumerate(mappings["item_ids"])}

    def _to_frame(self, item_idx: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
        frame = self.items.iloc[item_idx].copy()
        frame["score"] = np.round(scores, 4)
        frame["rank"] = np.arange(1, len(frame) + 1)
        return frame[OUTPUT_COLUMNS].reset_index(drop=True)

    def _item_indices(self, app_ids: list[int]) -> np.ndarray:
        unknown = [a for a in app_ids if a not in self.item_index]
        if unknown:
            raise ValueError(f"Games not in the catalogue: {unknown}")
        return np.array([self.item_index[a] for a in app_ids])

    def recommend_for_user(self, user_id: int, k: int = 10) -> pd.DataFrame:
        """Top-K unseen games for a user the model was trained on."""
        if user_id not in self.user_index:
            raise ValueError(f"Unknown user {user_id}; use recommend_for_new_user instead")
        idx = np.array([self.user_index[user_id]])
        recs, scores = self.model.recommend(self.history[idx], k=k, exclude=self.seen[idx], user_idx=idx)
        return self._to_frame(recs[0], scores[0])

    def recommend_for_new_user(self, liked_app_ids: list[int], k: int = 10) -> pd.DataFrame:
        """Cold start: builds a one-row history from games the user says they like."""
        cols = self._item_indices(liked_app_ids)
        history = sp.csr_matrix((np.ones(len(cols), dtype=np.float32), (np.zeros(len(cols)), cols)),
                                shape=(1, len(self.item_index)))
        recs, scores = self.model.recommend(history, k=k, exclude=history)
        return self._to_frame(recs[0], scores[0])

    def similar_games(self, app_id: int, k: int = 10) -> pd.DataFrame:
        """Games closest to `app_id` by tags and description."""
        recs, scores = self.similarity_model.similar_items(self._item_indices([app_id])[0], k=k)
        return self._to_frame(recs, scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get game recommendations")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--user-id", type=int, help="existing user id")
    group.add_argument("--liked", type=int, nargs="+", help="app ids a new user likes")
    group.add_argument("--similar-to", type=int, help="app id to find similar games for")
    parser.add_argument("-k", type=int, default=10, help="number of recommendations")
    args = parser.parse_args()

    pipeline = PredictionPipeline()
    if args.user_id is not None:
        result = pipeline.recommend_for_user(args.user_id, args.k)
    elif args.liked:
        result = pipeline.recommend_for_new_user(args.liked, args.k)
    else:
        result = pipeline.similar_games(args.similar_to, args.k)
    print(result.to_string(index=False))
