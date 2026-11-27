"""Runs the web app locally.

Usage:
    python -m app.main                 # http://127.0.0.1:5000
    python -m app.main --port 8000
"""
import argparse

from app import create_app

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve the game recommender on localhost")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    create_app().run(host="127.0.0.1", port=args.port, debug=args.debug)
