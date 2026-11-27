"""Downloads the raw Steam data, samples users and keeps a dense interaction core."""
import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.configuration.configuration import DataIngestionConfig
from src.constant import (DATE_COL, GAMES_FILE, HOURS_COL, INTERACTIONS_FILE, ITEM_COL, ITEMS_FILE,
                          KAGGLE_DATASET, METADATA_FILE, RAW_FILES, RECOMMENDATIONS_FILE, REVIEW_COL,
                          TARGET_COL, USER_COL)
from src.exception import CustomException
from src.logger import logger

INTERACTION_DTYPES = {ITEM_COL: "int32", USER_COL: "int32", REVIEW_COL: "int32",
                      TARGET_COL: "bool", HOURS_COL: "float32", DATE_COL: "string"}

ITEM_COLUMNS = [ITEM_COL, "title", "date_release", "rating", "positive_ratio", "user_reviews",
                "price_final", "win", "mac", "linux", "steam_deck", "tags", "description"]


@dataclass(frozen=True)
class DataIngestionArtifact:
    interactions_path: Path
    items_path: Path


def filter_sparse(df: pd.DataFrame, min_user: int, min_item: int, max_iter: int = 20) -> pd.DataFrame:
    """Drops rare users and games repeatedly until both thresholds hold (k-core)."""
    for _ in range(max_iter):
        user_counts = df[USER_COL].map(df[USER_COL].value_counts())
        item_counts = df[ITEM_COL].map(df[ITEM_COL].value_counts())
        keep = (user_counts >= min_user) & (item_counts >= min_item)
        if keep.all():
            break
        df = df[keep]
    return df.reset_index(drop=True)


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_raw_data(self) -> None:
        """Fetches the Kaggle dataset only when a raw file is missing."""
        raw_dir = self.config.raw_dir
        if all((raw_dir / name).exists() for name in RAW_FILES):
            logger.info(f"Raw data found in {raw_dir}")
            return

        import kagglehub

        logger.info(f"Downloading {KAGGLE_DATASET} from Kaggle")
        raw_dir.mkdir(parents=True, exist_ok=True)
        cache_path = Path(kagglehub.dataset_download(KAGGLE_DATASET))
        for item in cache_path.iterdir():
            if item.is_file():
                shutil.copy2(item, raw_dir / item.name)

    def load_interactions(self) -> pd.DataFrame:
        """Reads the 2 GB reviews file in chunks and keeps a deterministic user sample."""
        chunks = []
        reader = pd.read_csv(self.config.raw_dir / RECOMMENDATIONS_FILE, usecols=list(INTERACTION_DTYPES),
                             dtype=INTERACTION_DTYPES, chunksize=self.config.chunk_size)
        for chunk in reader:
            chunks.append(chunk[chunk[USER_COL] % self.config.user_sample_mod == 0])
        df = pd.concat(chunks, ignore_index=True)
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], format="%Y-%m-%d")
        logger.info(f"Sampled {len(df):,} interactions from {df[USER_COL].nunique():,} users")
        return df

    def load_items(self, app_ids: pd.Series) -> pd.DataFrame:
        """Joins game details with tags/descriptions for the games kept in the interactions."""
        games = pd.read_csv(self.config.raw_dir / GAMES_FILE)
        metadata = pd.read_json(self.config.raw_dir / METADATA_FILE, lines=True)
        items = games.merge(metadata, on=ITEM_COL, how="left")
        items = items[items[ITEM_COL].isin(app_ids)].copy()
        items["tags"] = items["tags"].apply(lambda t: list(t) if isinstance(t, list) else [])
        items["description"] = items["description"].fillna("")
        return items[ITEM_COLUMNS].reset_index(drop=True)

    def initiate_data_ingestion(self, force: bool = False) -> DataIngestionArtifact:
        processed_dir = self.config.processed_dir
        artifact = DataIngestionArtifact(processed_dir / INTERACTIONS_FILE, processed_dir / ITEMS_FILE)
        if not force and artifact.interactions_path.exists() and artifact.items_path.exists():
            logger.info("Processed data already exists, skipping ingestion")
            return artifact

        try:
            self.download_raw_data()
            interactions = filter_sparse(self.load_interactions(), self.config.min_user_interactions,
                                         self.config.min_item_interactions)
            items = self.load_items(interactions[ITEM_COL].drop_duplicates())

            # Keep only interactions whose game has catalogue details
            interactions = interactions[interactions[ITEM_COL].isin(items[ITEM_COL])]
            logger.info(f"After filtering: {len(interactions):,} interactions, "
                        f"{interactions[USER_COL].nunique():,} users, {len(items):,} games")

            processed_dir.mkdir(parents=True, exist_ok=True)
            interactions.to_parquet(artifact.interactions_path, index=False)
            items.to_parquet(artifact.items_path, index=False)
            logger.info(f"Saved processed data to {processed_dir}")
            return artifact
        except Exception as e:
            raise CustomException(e) from e
