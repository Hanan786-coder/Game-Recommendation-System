"""Common interface for every recommender: fit, score, recommend top-K."""
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import scipy.sparse as sp


def binarize(matrix: sp.csr_matrix) -> sp.csr_matrix:
    out = matrix.copy().astype(np.float32)
    out.data[:] = 1.0
    return out


def keep_top_neighbors(similarity: np.ndarray, n_neighbors: int, chunk_size: int = 1000) -> sp.csr_matrix:
    """Keeps the strongest `n_neighbors` per item (row) and drops self-similarity."""
    n_items = similarity.shape[0]
    n = min(n_neighbors, n_items - 1)
    np.fill_diagonal(similarity, 0.0)

    rows, cols, vals = [], [], []
    for start in range(0, n_items, chunk_size):  # chunked to limit memory
        block = similarity[start:start + chunk_size]
        top = np.argpartition(-block, n - 1, axis=1)[:, :n]
        rows.append(np.repeat(np.arange(start, start + len(block)), n))
        cols.append(top.ravel())
        vals.append(np.take_along_axis(block, top, axis=1).ravel())

    pruned = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                           shape=similarity.shape, dtype=np.float32)
    pruned.eliminate_zeros()
    return pruned


class BaseRecommender(ABC):
    name = "base"

    @abstractmethod
    def fit(self, interactions: sp.csr_matrix, items: pd.DataFrame | None = None) -> "BaseRecommender":
        """Learns from a user x item matrix of liked games (values = playtime strength)."""

    @abstractmethod
    def score(self, user_items: sp.csr_matrix, user_idx: np.ndarray | None = None) -> np.ndarray:
        """Returns a dense (n_users x n_items) score matrix for the given users' histories."""

    def get_params(self) -> dict:
        return dict(self.params)

    def recommend(self, user_items: sp.csr_matrix, k: int = 10, exclude: sp.csr_matrix | None = None,
                  user_idx: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Top-K item indices and scores per user, skipping items in `exclude`."""
        scores = np.array(self.score(user_items, user_idx), dtype=np.float32)
        if exclude is not None:
            rows, cols = exclude.nonzero()
            scores[rows, cols] = -np.inf

        k = min(k, scores.shape[1])
        top = np.argpartition(-scores, k - 1, axis=1)[:, :k]
        top_scores = np.take_along_axis(scores, top, axis=1)
        order = np.argsort(-top_scores, axis=1)  # argpartition is unordered
        return np.take_along_axis(top, order, axis=1), np.take_along_axis(top_scores, order, axis=1)

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{type(self).__name__}({params})"


class SimilarityRecommender(BaseRecommender):
    """Shared logic for item-to-item models: score = user history @ item similarity."""

    similarity: sp.csr_matrix

    def score(self, user_items, user_idx=None):
        return (binarize(user_items) @ self.similarity).toarray()

    def similar_items(self, item_idx: int, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        row = self.similarity[item_idx].toarray().ravel()
        top = np.argsort(-row)[:k]
        return top, row[top]
