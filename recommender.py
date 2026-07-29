"""Контентный рекомендатель товаров.

Представляет каждое описание товара как TF-IDF вектор и рекомендует
товары, чьи векторы ближе всего по косинусному сходству. Никаких данных о
пользователях, никаких оценок - просто "это описание похоже на то".
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


# Обычных английских стоп-слов тут недостаточно - карточки Flipkart делят
# между собой набор рекламных/маркетинговых слов почти в каждой категории
# ("buy", "online", "genuine", "shop"), которые иначе выглядели бы как
# значимое совпадение между никак не связанными товарами.
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
            max_df=0.6,        # выбросить термины, встречающиеся в >60% карточек - слишком общие, не помогают
            min_df=2,          # выбросить опечатки/разовые токены, встретившиеся только один раз
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
        # косинусное сходство одной строки со всей матрицей - позволяет не
        # строить полную матрицу сходства NxN, которая при таком размере
        # каталога весила бы примерно 1.5 ГБ чисел с плавающей точкой.
        scores = linear_kernel(query_vector, self.tfidf_matrix).flatten()

        # исключаем сам товар-запрос перед тем как брать топ результатов
        scores[idx] = -1
        top_indices = scores.argsort()[::-1][:top_n]

        results = self.products.iloc[top_indices].copy()
        results["similarity"] = scores[top_indices]
        return results.reset_index(drop=True)

    def explain(self, product_id, top_n=8):
        """Возвращает термины с наибольшим TF-IDF весом для товара - полезно
        чтобы понять *почему* модель считает два товара похожими (или
        не находит то, что ожидалось)."""
        idx = self.products.index[self.products["product_id"] == product_id][0]
        row = self.tfidf_matrix[idx].toarray().flatten()
        feature_names = self.vectorizer.get_feature_names_out()
        top_term_idx = row.argsort()[::-1][:top_n]
        return [(feature_names[i], round(row[i], 3)) for i in top_term_idx if row[i] > 0]
