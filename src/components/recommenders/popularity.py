"""Baseline: recommend the games liked by the most users."""
import numpy as np

from src.components.recommenders.base import BaseRecommender


class PopularityRecommender(BaseRecommender):
    name = "popularity"

    def __init__(self):
        self.params = {}
        self.item_counts = None

    def fit(self, interactions, items=None):
        self.item_counts = interactions.getnnz(axis=0).astype(np.float32)
        return self

    def score(self, user_items, user_idx=None):
        # Same ranking for every user; seen games are masked later
        return np.tile(self.item_counts, (user_items.shape[0], 1))
