import pandas as pd

from src.components.data_ingestion import filter_sparse
from src.components.data_transformation import build_matrices, temporal_split
from src.constant import DATE_COL, ITEM_COL, ITEM_IDX, TARGET_COL, USER_COL, USER_IDX


def test_split_holds_out_most_recent_interactions(interactions):
    train, val, test = temporal_split(interactions, val_per_user=1, test_per_user=1)

    assert len(train) + len(val) + len(test) == len(interactions)
    assert test.groupby(USER_IDX).size().max() == 1
    # Every held-out interaction is newer than the user's training history
    last_train = train.groupby(USER_IDX)[DATE_COL].max()
    assert (val.set_index(USER_IDX)[DATE_COL] >= last_train.loc[val[USER_IDX]].values).all()
    assert (test.set_index(USER_IDX)[DATE_COL] >= val.set_index(USER_IDX)[DATE_COL]).all()


def test_short_histories_stay_in_train(interactions):
    short = interactions[interactions[USER_IDX] == 0].head(2)
    train, val, test = temporal_split(short, val_per_user=1, test_per_user=1)
    assert len(train) == 2 and val.empty and test.empty


def test_matrices_keep_only_liked_games_as_positive(interactions):
    positive, seen = build_matrices(interactions, n_users=60, n_items=25)
    assert positive.nnz == interactions[TARGET_COL].sum()
    assert seen.nnz == len(interactions.drop_duplicates([USER_IDX, ITEM_IDX]))
    assert positive.data.min() >= 1.0


def test_filter_sparse_reaches_k_core():
    df = pd.DataFrame({USER_COL: [1, 1, 1, 2, 2, 3], ITEM_COL: [10, 11, 12, 10, 11, 10]})
    out = filter_sparse(df, min_user=2, min_item=2)
    assert out[USER_COL].value_counts().min() >= 2
    assert out[ITEM_COL].value_counts().min() >= 2
