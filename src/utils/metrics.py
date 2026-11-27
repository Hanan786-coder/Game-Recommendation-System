"""Top-K ranking metrics computed from a boolean hit matrix (users x K)."""
import numpy as np


def hit_matrix(recommendations: np.ndarray, relevant: np.ndarray) -> np.ndarray:
    """hits[u, r] is True when the r-th recommendation of user u is a relevant item."""
    return np.take_along_axis(relevant, recommendations, axis=1)


def ranking_metrics(hits: np.ndarray, n_relevant: np.ndarray) -> dict[str, np.ndarray]:
    """Per-user HitRate, Precision, Recall, NDCG and MAP at K = hits.shape[1]."""
    hits = hits.astype(np.float32)
    k = hits.shape[1]
    n_relevant = np.maximum(np.asarray(n_relevant, dtype=np.float32), 1.0)
    n_hits = hits.sum(axis=1)

    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (hits * discounts).sum(axis=1)
    ideal_len = np.minimum(n_relevant, k).astype(int)
    idcg = np.cumsum(discounts)[ideal_len - 1]

    # Average precision: precision at each hit position, averaged over the achievable hits
    precision_at_rank = np.cumsum(hits, axis=1) / np.arange(1, k + 1)
    ap = (precision_at_rank * hits).sum(axis=1) / np.minimum(n_relevant, k)

    return {
        "hit_rate": (n_hits > 0).astype(np.float32),
        "precision": n_hits / k,
        "recall": n_hits / n_relevant,
        "ndcg": dcg / idcg,
        "map": ap,
    }
