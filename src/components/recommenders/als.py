"""Matrix factorisation for implicit feedback (ALS, Hu et al. 2008) via the `implicit` library."""
import numpy as np
from threadpoolctl import threadpool_limits

from src.components.recommenders.base import BaseRecommender


class ALSRecommender(BaseRecommender):
    name = "als"

    def __init__(self, factors: int = 64, regularization: float = 0.05, iterations: int = 20,
                 alpha: float = 2.0, random_state: int = 42):
        self.params = {"factors": factors, "regularization": regularization, "iterations": iterations,
                       "alpha": alpha, "random_state": random_state}
        self.user_factors = None
        self.item_factors = None

    def fit(self, interactions, items=None):
        from implicit.als import AlternatingLeastSquares

        # Confidence grows with playtime; implicit already parallelises, so keep BLAS single-threaded
        confidence = (interactions * self.params["alpha"]).astype(np.float32).tocsr()
        with threadpool_limits(1, "blas"):
            model = AlternatingLeastSquares(
                factors=self.params["factors"],
                regularization=self.params["regularization"],
                iterations=self.params["iterations"],
                random_state=self.params["random_state"],
                use_gpu=False,
            )
            model.fit(confidence, show_progress=False)

        # Keep plain numpy arrays so the pickled model does not depend on implicit internals
        self.user_factors = np.asarray(model.user_factors, dtype=np.float32)
        self.item_factors = np.asarray(model.item_factors, dtype=np.float32)
        return self

    def score(self, user_items, user_idx=None):
        users = self.user_factors[user_idx] if user_idx is not None else self.fold_in(user_items)
        return users @ self.item_factors.T

    def fold_in(self, user_items) -> np.ndarray:
        """Solves the ALS user step for histories not seen in training (e.g. a new user)."""
        Y = self.item_factors
        base = Y.T @ Y + self.params["regularization"] * np.eye(Y.shape[1], dtype=np.float32)
        users = np.zeros((user_items.shape[0], Y.shape[1]), dtype=np.float32)

        for row in range(user_items.shape[0]):
            start, end = user_items.indptr[row], user_items.indptr[row + 1]
            if start == end:
                continue
            items = user_items.indices[start:end]
            conf = self.params["alpha"] * user_items.data[start:end]
            Yi = Y[items]
            A = base + (Yi.T * (conf - 1.0)) @ Yi
            b = (Yi * conf[:, None]).sum(axis=0)
            users[row] = np.linalg.solve(A, b)
        return users
