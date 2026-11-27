"""Small synthetic dataset shared by the tests."""
import numpy as np
import pandas as pd
import pytest

from src.components.data_transformation import InteractionData, temporal_split
from src.constant import DATE_COL, HOURS_COL, ITEM_IDX, REVIEW_COL, TARGET_COL, USER_IDX

N_USERS, N_ITEMS = 60, 25


@pytest.fixture
def interactions() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for user in range(N_USERS):
        items = rng.choice(N_ITEMS, size=rng.integers(3, 10), replace=False)
        for item in items:
            rows.append((user, item, rng.random() < 0.85, float(rng.exponential(20)),
                         pd.Timestamp("2020-01-01") + pd.Timedelta(days=int(rng.integers(0, 900)))))
    df = pd.DataFrame(rows, columns=[USER_IDX, ITEM_IDX, TARGET_COL, HOURS_COL, DATE_COL])
    df[REVIEW_COL] = np.arange(len(df))
    return df


@pytest.fixture
def items() -> pd.DataFrame:
    tags = [["Action", "Shooter"], ["RPG", "Open World"], ["Puzzle", "Indie"], ["Strategy"], ["Racing"]]
    return pd.DataFrame({
        ITEM_IDX: range(N_ITEMS),
        "tags": [tags[i % len(tags)] for i in range(N_ITEMS)],
        "description": [f"game number {i} with {' '.join(tags[i % len(tags)])} gameplay" for i in range(N_ITEMS)],
    })


@pytest.fixture
def data(interactions, items) -> InteractionData:
    train, val, test = temporal_split(interactions, val_per_user=1, test_per_user=1)
    return InteractionData(train, val, test, items, np.arange(N_USERS), np.arange(N_ITEMS))
