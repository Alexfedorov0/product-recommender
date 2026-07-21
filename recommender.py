"""Content-based product recommender.

Represents each product description as a TF-IDF vector and recommends
the products whose vectors are closest by cosine similarity. No user
data, no ratings - purely "this product reads like that product".
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


# Generic English stopwords aren't enough here - Flipkart listings share a
# handful of boilerplate/marketing words across almost every category
# ("buy", "online", "genuine", "shop") that would otherwise show up as
# meaningful overlap between unrelated products.
EXTRA_STOPWORDS = [
    "buy", "online", "genuine", "shop", "shopping", "india", "rs",
    "flipkart", "com", "branded", "best", "huge", "collection",
    "delivery", "guarantee", "replacement", "day", "free", "price",
    "cash", "products", "product",
]


class ContentBasedRecommender:
    def __init__(self, catalog):
        self.catalog = catalog
        self.products = catalog.products.reset_index(drop=True)
        self._fit()

    def _fit(self):
        descriptions = self.products["description"].tolist()
        self.vectorizer = TfidfVectorizer(
            stop_words=self._build_stopword_list(),
            max_df=0.6,        # drop terms that show up in >60% of listings - too generic to help
            min_df=2,          # drop typos/one-off tokens that only appear once
            ngram_range=(1, 2),
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(descriptions)

    @staticmethod
    def _build_stopword_list():
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        return list(ENGLISH_STOP_WORDS.union(EXTRA_STOPWORDS))

    def recommend(self, product_id, top_n=5):
        idx = self.products.index[self.products["product_id"] == product_id]
        if len(idx) == 0:
            raise KeyError(f"No product with id {product_id}")
        idx = idx[0]

        query_vector = self.tfidf_matrix[idx]
        # cosine similarity between one row and the whole matrix - avoids
        # ever materializing the full NxN similarity matrix, which would be
        # ~1.5GB of floats at this catalog size.
        scores = linear_kernel(query_vector, self.tfidf_matrix).flatten()

        # exclude the query product itself before taking the top results
        scores[idx] = -1
        top_indices = scores.argsort()[::-1][:top_n]

        results = self.products.iloc[top_indices].copy()
        results["similarity"] = scores[top_indices]
        return results.reset_index(drop=True)

    def explain(self, product_id, top_n=8):
        """Returns the highest-weighted TF-IDF terms for a product - useful
        for understanding *why* the model thinks two products are similar
        (or isn't finding what you'd expect)."""
        idx = self.products.index[self.products["product_id"] == product_id][0]
        row = self.tfidf_matrix[idx].toarray().flatten()
        feature_names = self.vectorizer.get_feature_names_out()
        top_term_idx = row.argsort()[::-1][:top_n]
        return [(feature_names[i], round(row[i], 3)) for i in top_term_idx if row[i] > 0]
