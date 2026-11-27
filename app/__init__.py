"""Flask web app that serves the trained game recommender."""
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from app.service import RecommenderService
from src.constant import ROOT_DIR

MAX_K = 20


def create_app(artifacts_dir: Path = ROOT_DIR / "artifacts") -> Flask:
    app = Flask(__name__)
    service = RecommenderService(artifacts_dir)  # loaded once at start-up

    def error(message: str, status: int = 400):
        return jsonify({"error": message}), status

    def read_k(default: int = 10) -> int:
        k = request.args.get("k", default, type=int) or default  # invalid values fall back to default
        return max(1, min(k, MAX_K))

    @app.get("/")
    def index():
        return render_template("index.html", info=service.model_info())

    @app.get("/api/search")
    def search():
        return jsonify(service.search(request.args.get("q", "")))

    @app.get("/api/popular")
    def popular():
        return jsonify(service.popular())

    @app.get("/api/sample-users")
    def sample_users():
        return jsonify(service.sample_users())

    @app.post("/api/recommendations")
    def recommend_for_games():
        payload = request.get_json(silent=True) or {}
        app_ids = payload.get("app_ids") or []
        if not isinstance(app_ids, list) or not app_ids:
            return error("Pick at least one game to get recommendations.")
        try:
            return jsonify({"results": service.recommend_for_games([int(a) for a in app_ids], read_k())})
        except ValueError as e:
            return error(str(e))

    @app.get("/api/users/<int:user_id>/recommendations")
    def recommend_for_user(user_id: int):
        try:
            return jsonify(service.recommend_for_user(user_id, read_k()))
        except ValueError:
            return error(f"No player with ID {user_id} in the training data. Try one of the sample IDs.", 404)

    @app.get("/api/games/<int:app_id>/similar")
    def similar(app_id: int):
        try:
            return jsonify(service.similar(app_id, read_k(8)))
        except ValueError:
            return error(f"Game {app_id} is not in the catalogue.", 404)

    @app.get("/api/model")
    def model():
        return jsonify(service.model_info())

    return app
