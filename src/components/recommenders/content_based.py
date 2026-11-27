"""Content-based filtering: games with similar tags and descriptions are similar."""
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from src.components.recommenders.base import SimilarityRecommender, keep_top_neighbors


def tags_to_text(tags) -> str:
    # "Open World" -> "open_world" so multi-word tags stay a single token
    return " ".join(str(tag).lower().replace(" ", "_") for tag in tags)


class ContentBasedRecommender(SimilarityRecommender):
    name = "content"

    def __init__(self, n_neighbors: int = 100, tag_weight: float = 1.0):
        self.params = {"n_neighbors": n_neighbors, "tag_weight": tag_weight}
        self.similarity = None

    def fit(self, interactions, items=None):
        if items is None:
            raise ValueError("ContentBasedRecommender needs the items table (tags, description)")

        tag_tfidf = TfidfVectorizer(token_pattern=r"\S+").fit_transform(items["tags"].apply(tags_to_text))
        desc_tfidf = TfidfVectorizer(stop_words="english", min_df=2, max_features=20000,
                                     sublinear_tf=True).fit_transform(items["description"])

        # Weighted tags + description, L2-normalised so the dot product is cosine similarity
        features = normalize(sp.hstack([self.params["tag_weight"] * tag_tfidf, desc_tfidf]).tocsr())
        sim = (features @ features.T).toarray().astype(np.float32)
        self.similarity = keep_top_neighbors(sim, self.params["n_neighbors"])
        return self
