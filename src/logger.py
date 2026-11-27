"""Project logger: writes to logs/<timestamp>.log and to the console."""
import logging
import sys
from datetime import datetime

from src.constant import ROOT_DIR

LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"{datetime.now():%Y_%m_%d_%H_%M_%S}.log"

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(module)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger("game_recommender")
