"""Web app tests on a tiny set of artifacts built from the synthetic fixture data."""
import pytest
import scipy.sparse as sp

from app import create_app
from src.components.recommenders import ContentBasedRecommender, ItemKNNRecommender
from src.constant import HISTORY_FILE, ITEMS_FILE, MAPPINGS_FILE, MODEL_FILE, SEEN_FILE, SIMILARITY_MODEL_FILE
from src.utils.common import save_object

APP_ID_OFFSET = 1000  # fake Steam ids, distinct from matrix indices


@pytest.fixture
def client(data, tmp_path):
    items = data.items.copy()
    items["app_id"] = items["item_idx"] + APP_ID_OFFSET
    items["title"] = [f"Game {i}" for i in range(len(items))]
    items["date_release"] = "2020-01-01"
    items["rating"] = "Very Positive"
    items["positive_ratio"] = 90
    items["price_final"] = 9.99

    history, seen = data.matrices(data.train, data.val, data.test)
    save_object(ItemKNNRecommender(n_neighbors=5).fit(history, items), tmp_path / MODEL_FILE)
    save_object({"content": ContentBasedRecommender(n_neighbors=5).fit(history, items),
                 "collaborative": ItemKNNRecommender(n_neighbors=5).fit(history, items)},
                tmp_path / SIMILARITY_MODEL_FILE)
    save_object({"user_ids": data.user_ids, "item_ids": items["app_id"].to_numpy()}, tmp_path / MAPPINGS_FILE)
    items.to_parquet(tmp_path / ITEMS_FILE, index=False)
    sp.save_npz(tmp_path / HISTORY_FILE, history)
    sp.save_npz(tmp_path / SEEN_FILE, seen)

    return create_app(tmp_path).test_client()


def test_home_page_renders(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"What should you play next?" in response.data


def test_search_matches_titles(client):
    results = client.get("/api/search?q=game 1").get_json()
    assert results and all("game 1" in r["title"].lower() for r in results)
    assert client.get("/api/search?q=g").get_json() == []


def test_recommendations_for_picked_games(client):
    picked = [APP_ID_OFFSET + 1, APP_ID_OFFSET + 2]
    results = client.post("/api/recommendations", json={"app_ids": picked}).get_json()["results"]
    assert [r["rank"] for r in results] == list(range(1, len(results) + 1))
    assert not {r["app_id"] for r in results} & set(picked)


def test_recommendations_need_a_game(client):
    response = client.post("/api/recommendations", json={"app_ids": []})
    assert response.status_code == 400 and "error" in response.get_json()


def test_unknown_game_is_rejected(client):
    assert client.post("/api/recommendations", json={"app_ids": [1]}).status_code == 400
    assert client.get("/api/games/1/similar").status_code == 404


def test_known_and_unknown_players(client):
    body = client.get("/api/users/0/recommendations?k=5").get_json()
    assert body["user_id"] == 0 and len(body["results"]) == 5 and body["liked"]
    assert client.get("/api/users/999999/recommendations").status_code == 404


def test_similar_games(client):
    body = client.get(f"/api/games/{APP_ID_OFFSET}/similar").get_json()
    assert body["game"]["app_id"] == APP_ID_OFFSET
    assert all(r["app_id"] != APP_ID_OFFSET for r in body["results"])
