from setuptools import find_packages, setup


def get_requirements(path: str = "requirements.txt") -> list[str]:
    # Skip the editable-install line used for local development
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("-e")]


setup(
    name="game-recommender",
    version="0.1.0",
    description="Top-N game recommendation system trained on Steam reviews",
    packages=find_packages(exclude=["tests", "tests.*"]),
    python_requires=">=3.10",
    install_requires=get_requirements(),
)
