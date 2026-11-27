"""Thin layer between the web routes and the trained model: search, recommend, explain."""
from pathlib import Path

import numpy as np

from src.constant import EVALUATION_REPORT_FILE, ROOT_DIR
from src.pipeline.prediction_pipeline import PredictionPipeline
from src.utils.common import load_json

COVER_URL = "https://cdn.cloudflare.steamstatic.com/steam/apps/{app_id}/header.jpg"
MODEL_NAMES = {"item_knn": "Item-KNN", "als": "ALS", "content": "Content-based", "popularity": "Popularity"}


class RecommenderService:
    def __init__(self, artifacts_dir: Path = ROOT_DIR / "artifacts"):
        self.pipeline = PredictionPipeline(artifacts_dir)
        self.items = self.pipeline.items
        self.titles = self.items["title"].str.casefold().to_numpy().astype(str)
        self.popularity = self.pipeline.history.getnnz(axis=0)  # likes per game, used to rank search hits
        report_path = artifacts_dir / EVALUATION_REPORT_FILE
        self.report = load_json(report_path) if report_path.exists() else None

    # ---- catalogue ----------------------------------------------------------------------------

    def game(self, item_idx: int) -> dict:
        row = self.items.iloc[int(item_idx)]
        return {
            "app_id": int(row["app_id"]),
            "title": str(row["title"]),
            "year": str(row["date_release"])[:4],
            "rating": str(row["rating"]),
            "positive_ratio": int(row["positive_ratio"]),
            "price": float(row["price_final"]),
            "tags": [str(t) for t in list(row["tags"])[:3]],
            "cover": COVER_URL.format(app_id=int(row["app_id"])),
        }

    def search(self, query: str, limit: int = 8) -> list[dict]:
        query = query.strip().casefold()
        if len(query) < 2:
            return []
        matches = np.flatnonzero(np.char.find(self.titles, query) >= 0)
        # Titles that start with the query first, then the most-liked games
        starts = np.char.startswith(self.titles[matches], query)
        order = np.lexsort((-self.popularity[matches], ~starts))
        return [self.game(i) for i in matches[order][:limit]]

    def popular(self, limit: int = 12) -> list[dict]:
        return [self.game(i) for i in np.argsort(-self.popularity)[:limit]]

    def sample_users(self, n: int = 4, seed: int = 7) -> list[int]:
        """A few real users with a readable history, for the demo."""
        likes = self.pipeline.history.getnnz(axis=1)
        candidates = np.flatnonzero((likes >= 5) & (likes <= 12))
        picked = np.random.default_rng(seed).choice(candidates, size=min(n, len(candidates)), replace=False)
        user_ids = np.array(list(self.pipeline.user_index))
        return [int(user_ids[i]) for i in picked]

    # ---- recommendations ----------------------------------------------------------------------

    def _because(self, liked: np.ndarray, recs: np.ndarray) -> list[str | None]:
        """For item-similarity models, the liked game that contributed most to each pick."""
        similarity = getattr(self.pipeline.model, "similarity", None)
        if similarity is None or len(liked) == 0:
            return [None] * len(recs)
        contrib = similarity[liked][:, recs].toarray()
        best = contrib.argmax(axis=0)
        return [str(self.items.iloc[liked[b]]["title"]) if contrib[b, j] > 0 else None
                for j, b in enumerate(best)]

    def _results(self, recs: np.ndarray, liked: np.ndarray) -> list[dict]:
        because = self._because(liked, recs)
        return [{**self.game(idx), "rank": rank, "because": because[rank - 1]}
                for rank, idx in enumerate(recs, start=1)]

    def recommend_for_games(self, app_ids: list[int], k: int = 10) -> list[dict]:
        frame = self.pipeline.recommend_for_new_user(app_ids, k)
        recs = np.array([self.pipeline.item_index[a] for a in frame["app_id"]])
        return self._results(recs, self.pipeline._item_indices(app_ids))

    def recommend_for_user(self, user_id: int, k: int = 10) -> dict:
        frame = self.pipeline.recommend_for_user(user_id, k)
        row = self.pipeline.history[self.pipeline.user_index[user_id]]
        liked = row.indices[np.argsort(-row.data)]  # most-played first
        recs = np.array([self.pipeline.item_index[a] for a in frame["app_id"]])
        return {"user_id": user_id, "liked": [self.game(i) for i in liked],
                "results": self._results(recs, liked)}

    def similar(self, app_id: int, k: int = 8) -> dict:
        frame = self.pipeline.similar_games(app_id, k)
        return {"game": self.game(self.pipeline.item_index[app_id]),
                "results": [self.game(self.pipeline.item_index[a]) for a in frame["app_id"]]}

    def model_info(self) -> dict:
        info = {"games": len(self.items), "players": len(self.pipeline.user_index)}
        if self.report:
            selected = self.report["selected_model"]
            info.update(model=MODEL_NAMES.get(selected, selected), k=self.report["k"],
                        metrics=self.report["models"][selected]["metrics"],
                        baseline=self.report["models"].get("popularity", {}).get("metrics"))
        return info
