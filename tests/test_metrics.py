import numpy as np
import pytest

from src.utils.metrics import hit_matrix, ranking_metrics


def test_hit_matrix_marks_relevant_positions():
    recs = np.array([[2, 0, 1]])
    relevant = np.array([[True, False, False, True]])
    assert hit_matrix(recs, relevant).tolist() == [[False, True, False]]


def test_perfect_ranking_scores_one():
    m = ranking_metrics(np.array([[True, True, False]]), n_relevant=[2])
    for name in ("hit_rate", "recall", "ndcg", "map"):
        assert m[name][0] == pytest.approx(1.0)
    assert m["precision"][0] == pytest.approx(2 / 3)


def test_single_hit_at_second_position():
    m = ranking_metrics(np.array([[False, True, False]]), n_relevant=[1])
    assert m["ndcg"][0] == pytest.approx(1 / np.log2(3))
    assert m["map"][0] == pytest.approx(0.5)
    assert m["recall"][0] == pytest.approx(1.0)


def test_no_hits_scores_zero():
    m = ranking_metrics(np.zeros((1, 5), dtype=bool), n_relevant=[3])
    assert all(values[0] == 0 for values in m.values())
