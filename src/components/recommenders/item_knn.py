"""Item-based collaborative filtering: games liked by the same users are similar."""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.components.recommenders.base import SimilarityRecommender, binarize, keep_top_neighbors


class ItemKNNRecommender(SimilarityRecommender):
    name = "item_knn"

    def __init__(self, n_neighbors: int = 100):
        self.params = {"n_neighbors": n_neighbors}
        self.similarity = None

    def fit(self, interactions, items=None):
        # Columns are games, so cosine over the transposed matrix compares games by their users
        sim = cosine_similarity(binarize(interactions).T.tocsr(), dense_output=True).astype(np.float32)
        self.similarity = keep_top_neighbors(sim, self.params["n_neighbors"])
        return self
