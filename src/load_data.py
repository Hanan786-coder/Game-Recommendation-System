"""Download the Steam game recommendations dataset from Kaggle.

Usage:
    python src/data/load_data.py                      # saves to data/raw
    python src/data/load_data.py --output-dir /path   # custom location
    DATA_DIR=/path python src/data/load_data.py       # via env variable
"""
import argparse
import os
import shutil
from pathlib import Path

import kagglehub

DATASET = "antonkozyriev/game-recommendations-on-steam"
DEFAULT_DIR = Path(os.getenv("DATA_DIR", "data/raw"))


def download_dataset(output_dir: Path = DEFAULT_DIR, force: bool = False) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    if any(output_dir.iterdir()) and not force:
        print(f"Data already exists in {output_dir}, skipping (use --force to redownload).")
        return output_dir

    cache_path = Path(kagglehub.dataset_download(DATASET))

    for item in cache_path.iterdir():
        dest = output_dir / item.name
        if item.is_file():
            shutil.copy2(item, dest)

    print(f"Saved to {output_dir}: {sorted(p.name for p in output_dir.iterdir())}")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    download_dataset(args.output_dir, args.force)