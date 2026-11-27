# Game Recommendation System

A top-N recommender for Steam games. Given a user's review history, it ranks the games they are most likely to
enjoy. It can also recommend for a brand-new user from a few liked games and find games similar to a given title.

Built on the Kaggle dataset
[Game Recommendations on Steam](https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam)
(~41M reviews, ~14M users, ~50k games).

## Problem Framing

| | |
|---|---|
| **Type** | Top-N ranking with implicit feedback |
| **Input** | A user's reviewed games (thumbs up/down + hours played), or a list of liked games for a new user |
| **Output** | Ranked list of K games the user has not reviewed yet |
| **Signal** | Liked games are positives, weighted by playtime as `1 + log1p(hours)` |
| **Split** | Per-user temporal: most recent review → test, previous → validation, rest → train |
| **Metrics** | HitRate, Precision, Recall, NDCG, MAP @10, plus catalogue coverage |

To keep training on a laptop fast, the pipeline uses a deterministic 5% sample of users (`user_id % 20 == 0`). It
then keeps users with ≥3 reviews and games with ≥20 reviews. That leaves 1.33M interactions from 182,790 users
on 5,711 games.

## Models

| Model | Idea |
|---|---|
| Popularity | Most-liked games overall (baseline) |
| Item-KNN | Cosine similarity between games based on who liked them |
| Content-based | TF-IDF over tags and descriptions; also used for "similar games" |
| ALS | Implicit-feedback matrix factorisation (`implicit` library) |

Each model is grid-searched on the validation split. The model with the best validation NDCG@10 is selected and
reported on the test split. It is then retrained on all interactions and saved for serving.

## Results (test split, K = 10, 155,632 users)

| Model | NDCG | HitRate | MAP | Coverage |
|---|---|---|---|---|
| **Item-KNN** (selected) | **0.0539** | **0.0985** | **0.0404** | 0.971 |
| ALS | 0.0522 | 0.0963 | 0.0389 | 0.192 |
| Popularity | 0.0205 | 0.0426 | 0.0138 | 0.005 |
| Content-based | 0.0106 | 0.0181 | 0.0083 | 0.893 |

Item-KNN is about 2.6x the popularity baseline. ALS is close on accuracy but recommends from a much smaller part of
the catalogue. See [`notebooks/02_model_experiments.ipynb`](notebooks/02_model_experiments.ipynb) for the analysis.

## Project Structure

```
├── .github/workflows/tests.yml   # CI: runs the test suite
├── artifacts/                    # trained models, mappings, reports (generated)
├── data/
│   ├── raw/                      # Kaggle files (downloaded)
│   └── processed/                # sampled interactions, items, splits (generated)
├── notebooks/
│   ├── 01_eda.ipynb              # exploratory data analysis
│   └── 02_model_experiments.ipynb
├── src/
│   ├── components/
│   │   ├── data_ingestion.py     # download, user sampling, k-core filtering
│   │   ├── data_transformation.py# id mapping, temporal split, sparse matrices
│   │   ├── model_trainer.py      # grid search and refitting
│   │   ├── model_evaluation.py   # top-K evaluation and test report
│   │   └── recommenders/         # popularity, item-KNN, content-based, ALS
│   ├── configuration/            # config.yaml + typed config loader
│   ├── constant/                 # paths, file and column names
│   ├── pipeline/
│   │   ├── training_pipeline.py
│   │   └── prediction_pipeline.py
│   ├── utils/                    # I/O helpers and ranking metrics
│   ├── logger.py
│   └── exception.py
├── tests/
├── requirements.txt
└── setup.py
```

## Getting Started

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (source .venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
```

Train end to end. Raw data is downloaded from Kaggle if `data/raw` is empty; this needs Kaggle credentials for
`kagglehub`.

```bash
python -m src.pipeline.training_pipeline
```

The run takes about 4 minutes on a laptop and writes `artifacts/evaluation_report.json` and `artifacts/tuning_results.json`.
All settings (sample size, filters, model grids) live in `src/configuration/config.yaml`.

Get recommendations:

```bash
python -m src.pipeline.prediction_pipeline --user-id 51580            # existing user
python -m src.pipeline.prediction_pipeline --liked 367520 105600      # new user (Hollow Knight, Terraria)
python -m src.pipeline.prediction_pipeline --similar-to 413150        # games like Stardew Valley
```

Run the tests:

```bash
pytest -q
```

## Notes and Limitations

- Each user's last review is used as the test target, but other users' later reviews can still be in training. This
  is a common simplification; a global time cut-off would be stricter.
- About 10% of games have no tags or description in the source data. For those, "similar games" falls back to co-play
  similarity.
- Some titles contain broken characters (e.g. `DARK SOULS� III`) because they are stored that way in the raw files.
