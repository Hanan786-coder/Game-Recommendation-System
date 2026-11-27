"""Maps ids to indices, splits each user's history by time and builds sparse matrices."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.configuration.configuration import DataTransformationConfig
from src.constant import (DATE_COL, HOURS_COL, INTERACTIONS_FILE, ITEM_COL, ITEM_IDX, ITEMS_FILE, MAPPINGS_FILE,
                          REVIEW_COL, TARGET_COL, TEST_FILE, TRAIN_FILE, USER_COL, USER_IDX, VAL_FILE)
from src.exception import CustomException
from src.logger import logger
from src.utils.common import load_object, save_object

SPLIT_COLUMNS = [USER_IDX, ITEM_IDX, TARGET_COL, HOURS_COL, DATE_COL, REVIEW_COL]


def temporal_split(df: pd.DataFrame, val_per_user: int, test_per_user: int):
    """Holds out each user's most recent interactions: last -> test, previous -> validation."""
    df = df.sort_values([USER_IDX, DATE_COL, REVIEW_COL])
    recency = df.groupby(USER_IDX).cumcount(ascending=False)  # 0 = most recent
    n_user = df.groupby(USER_IDX)[ITEM_IDX].transform("size")

    # Users without enough history stay entirely in train
    eligible = n_user > val_per_user + test_per_user
    is_test = eligible & (recency < test_per_user)
    is_val = eligible & (recency >= test_per_user) & (recency < test_per_user + val_per_user)

    train = df[~(is_test | is_val)].reset_index(drop=True)
    return train, df[is_val].reset_index(drop=True), df[is_test].reset_index(drop=True)


def build_matrices(df: pd.DataFrame, n_users: int, n_items: int) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Returns (positive, seen) user x item matrices.

    positive: liked games only, weighted by playtime as 1 + log1p(hours).
    seen:     every reviewed game (liked or not), used to mask recommendations.
    """
    shape = (n_users, n_items)
    pos = df[df[TARGET_COL]]
    strength = (1.0 + np.log1p(pos[HOURS_COL].clip(lower=0))).astype(np.float32)
    positive = sp.csr_matrix((strength, (pos[USER_IDX], pos[ITEM_IDX])), shape=shape)
    seen = sp.csr_matrix((np.ones(len(df), dtype=np.float32), (df[USER_IDX], df[ITEM_IDX])), shape=shape)
    seen.data[:] = 1.0  # duplicates collapse to a single flag
    return positive, seen


@dataclass
class InteractionData:
    """Train/validation/test interactions plus the catalogue, all in index space."""
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    items: pd.DataFrame
    user_ids: np.ndarray
    item_ids: np.ndarray

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    @property
    def n_items(self) -> int:
        return len(self.item_ids)

    def matrices(self, *frames: pd.DataFrame) -> tuple[sp.csr_matrix, sp.csr_matrix]:
        return build_matrices(pd.concat(frames, ignore_index=True), self.n_users, self.n_items)


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def initiate_data_transformation(self) -> InteractionData:
        try:
            interactions = pd.read_parquet(self.config.processed_dir / INTERACTIONS_FILE)
            items = pd.read_parquet(self.config.processed_dir / ITEMS_FILE)

            # Contiguous indices for the matrix rows (users) and columns (games)
            user_ids = np.sort(interactions[USER_COL].unique())
            item_ids = np.sort(items[ITEM_COL].unique())
            interactions[USER_IDX] = np.searchsorted(user_ids, interactions[USER_COL]).astype(np.int32)
            interactions[ITEM_IDX] = np.searchsorted(item_ids, interactions[ITEM_COL]).astype(np.int32)
            items[ITEM_IDX] = np.searchsorted(item_ids, items[ITEM_COL])
            items = items.sort_values(ITEM_IDX).reset_index(drop=True)

            train, val, test = temporal_split(interactions[SPLIT_COLUMNS], self.config.val_per_user,
                                              self.config.test_per_user)
            logger.info(f"Split sizes - train: {len(train):,}, val: {len(val):,}, test: {len(test):,}")

            data = InteractionData(train, val, test, items, user_ids, item_ids)
            self.save(data)
            return data
        except Exception as e:
            raise CustomException(e) from e

    def save(self, data: InteractionData) -> None:
        processed_dir, artifacts_dir = self.config.processed_dir, self.config.artifacts_dir
        data.train.to_parquet(processed_dir / TRAIN_FILE, index=False)
        data.val.to_parquet(processed_dir / VAL_FILE, index=False)
        data.test.to_parquet(processed_dir / TEST_FILE, index=False)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        data.items.to_parquet(artifacts_dir / ITEMS_FILE, index=False)
        save_object({"user_ids": data.user_ids, "item_ids": data.item_ids}, artifacts_dir / MAPPINGS_FILE)
        logger.info(f"Saved splits to {processed_dir} and mappings to {artifacts_dir}")

    @staticmethod
    def load(processed_dir: Path, artifacts_dir: Path) -> InteractionData:
        """Reloads a previously saved transformation (used by notebooks)."""
        mappings = load_object(artifacts_dir / MAPPINGS_FILE)
        return InteractionData(
            train=pd.read_parquet(processed_dir / TRAIN_FILE),
            val=pd.read_parquet(processed_dir / VAL_FILE),
            test=pd.read_parquet(processed_dir / TEST_FILE),
            items=pd.read_parquet(artifacts_dir / ITEMS_FILE),
            user_ids=mappings["user_ids"],
            item_ids=mappings["item_ids"],
        )
