import numpy as np
import pytest
import scipy.sparse as sp

from src.components.model_evaluation import evaluate_model, users_with_targets
from src.components.model_trainer import param_grid
from src.components.recommenders import MODEL_REGISTRY, ALSRecommender, build_model

FAST_PARAMS = {"als": {"factors": 8, "iterations": 3}, "item_knn": {"n_neighbors": 5},
               "content": {"n_neighbors": 5}, "popularity": {}}


@pytest.mark.parametrize("name", list(MODEL_REGISTRY))
def test_recommend_returns_k_unseen_items(name, data):
    history, seen = data.matrices(data.train)
    model = build_model(name, **FAST_PARAMS[name]).fit(history, data.items)

    users = np.arange(10)
    recs, scores = model.recommend(history[users], k=5, exclude=seen[users], user_idx=users)

    assert recs.shape == (10, 5)
    assert np.all(np.diff(scores, axis=1) <= 1e-6)  # sorted best first
    for row, user in enumerate(users):
        assert not set(recs[row]) & set(seen[user].indices)


@pytest.mark.parametrize("name", list(MODEL_REGISTRY))
def test_evaluate_model_returns_bounded_metrics(name, data):
    history, seen = data.matrices(data.train)
    target, _ = data.matrices(data.val)
    model = build_model(name, **FAST_PARAMS[name]).fit(history, data.items)

    metrics = evaluate_model(model, history, seen, target, users_with_targets(target), k=5)
    assert set(metrics) == {"hit_rate", "precision", "recall", "ndcg", "map", "coverage"}
    assert all(0.0 <= v <= 1.0 for v in metrics.values())


def test_als_fold_in_matches_new_user_shape(data):
    history, _ = data.matrices(data.train)
    model = ALSRecommender(factors=8, iterations=3).fit(history)
    new_user = sp.csr_matrix(([1.0, 1.0], ([0, 0], [1, 2])), shape=(1, history.shape[1]))
    assert model.score(new_user).shape == (1, history.shape[1])


def test_similar_items_skips_games_without_metadata(data):
    items = data.items.copy()
    items.at[0, "tags"], items.at[0, "description"] = [], ""
    history, _ = data.matrices(data.train)
    model = build_model("content", n_neighbors=5).fit(history, items)

    neighbors, scores = model.similar_items(0, k=5)
    assert len(neighbors) == 0
    neighbors, scores = model.similar_items(1, k=5)
    assert len(neighbors) > 0 and np.all(scores > 0)


def test_param_grid_expands_combinations():
    spec = {"params": {"a": 1, "b": 2}, "grid": {"b": [3, 4], "c": [5]}}
    assert param_grid(spec) == [{"a": 1, "b": 3, "c": 5}, {"a": 1, "b": 4, "c": 5}]
    assert param_grid({"params": {}}) == [{}]


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        build_model("does_not_exist")
