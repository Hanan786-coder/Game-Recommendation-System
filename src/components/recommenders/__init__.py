"""Recommender models and a small registry to build them by name from the config."""
from src.components.recommenders.als import ALSRecommender
from src.components.recommenders.base import BaseRecommender, SimilarityRecommender
from src.components.recommenders.content_based import ContentBasedRecommender
from src.components.recommenders.item_knn import ItemKNNRecommender
from src.components.recommenders.popularity import PopularityRecommender

MODEL_REGISTRY = {cls.name: cls for cls in
                  (PopularityRecommender, ItemKNNRecommender, ContentBasedRecommender, ALSRecommender)}


def build_model(name: str, **params) -> BaseRecommender:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**params)


__all__ = ["BaseRecommender", "SimilarityRecommender", "PopularityRecommender", "ItemKNNRecommender",
           "ContentBasedRecommender", "ALSRecommender", "MODEL_REGISTRY", "build_model"]
