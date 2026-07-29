"""Быстрая демка: берём несколько товаров, смотрим что предлагает рекомендатель."""

from pathlib import Path

from product_catalog import ProductCatalog
from recommender import ContentBasedRecommender

# путь строится относительно расположения этого файла, а не текущей рабочей
# директории терминала - иначе будет "file not found" при запуске скрипта
# из другого места на диске.
DATA_PATH = Path(__file__).parent / "flipkart_com-ecommerce_sample.csv"


def print_recommendations(catalog, rec, product_id):
    product = catalog.get_product(product_id)
    print(f"\nЗАПРОС [{product['category']}]: {product['product_name']}")
    print(f"  {product['description'][:140]}...")
    print("  Рекомендации:")
    for _, r in rec.recommend(product_id, top_n=5).iterrows():
        print(f"    {r['similarity']:.3f}  [{r['category']:22s}] {r['product_name'][:55]}")


def main():
    catalog = ProductCatalog(DATA_PATH)
    print(f"Загружено {len(catalog)} товаров после очистки")

    rec = ContentBasedRecommender(catalog)
    print(f"Размер словаря TF-IDF: {len(rec.vectorizer.get_feature_names_out())}")

    # пара обычных примеров сходства внутри категории
    sample = catalog.sample(3, random_state=42)
    for pid in sample["product_id"]:
        print_recommendations(catalog, rec, pid)

    # случай с горошком на ткани - почему это интересно, см. README
    polka_top = catalog.products[
        catalog.products["product_name"].str.contains("Gypsy Soul Casual", na=False)
    ]
    if not polka_top.empty:
        print_recommendations(catalog, rec, polka_top.iloc[0]["product_id"])


if __name__ == "__main__":
    main()
