"""Project-wide constants: paths, file names and column names."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_FILE_PATH = ROOT_DIR / "src" / "configuration" / "config.yaml"

KAGGLE_DATASET = "antonkozyriev/game-recommendations-on-steam"

# Raw files shipped with the Kaggle dataset
GAMES_FILE = "games.csv"
USERS_FILE = "users.csv"
RECOMMENDATIONS_FILE = "recommendations.csv"
METADATA_FILE = "games_metadata.json"
RAW_FILES = (GAMES_FILE, USERS_FILE, RECOMMENDATIONS_FILE, METADATA_FILE)

# Processed data
INTERACTIONS_FILE = "interactions.parquet"
ITEMS_FILE = "items.parquet"
TRAIN_FILE = "train.parquet"
VAL_FILE = "val.parquet"
TEST_FILE = "test.parquet"

# Artifacts
MAPPINGS_FILE = "mappings.pkl"
MODEL_FILE = "model.pkl"
SIMILARITY_MODEL_FILE = "similarity_model.pkl"
HISTORY_FILE = "user_history.npz"
SEEN_FILE = "user_seen.npz"
TUNING_REPORT_FILE = "tuning_results.json"
EVALUATION_REPORT_FILE = "evaluation_report.json"

# Column names
USER_COL = "user_id"
ITEM_COL = "app_id"
USER_IDX = "user_idx"
ITEM_IDX = "item_idx"
TARGET_COL = "is_recommended"
HOURS_COL = "hours"
DATE_COL = "date"
REVIEW_COL = "review_id"
