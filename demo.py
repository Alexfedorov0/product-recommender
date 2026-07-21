"""Quick demo: pick a few products, show what the recommender suggests."""

from pathlib import Path

from product_catalog import ProductCatalog
from recommender import ContentBasedRecommender

# resolve relative to this file's location, not the terminal's current
# working directory - avoids "file not found" when the script is launched
# from somewhere else on disk.
DATA_PATH = Path(__file__).parent / "flipkart_com-ecommerce_sample.csv"


def print_recommendations(catalog, rec, product_id):
    product = catalog.get_product(product_id)
    print(f"\nQUERY [{product['category']}]: {product['product_name']}")
    print(f"  {product['description'][:140]}...")
    print("  Recommended:")
    for _, r in rec.recommend(product_id, top_n=5).iterrows():
        print(f"    {r['similarity']:.3f}  [{r['category']:22s}] {r['product_name'][:55]}")


def main():
    catalog = ProductCatalog(DATA_PATH)
    print(f"Loaded {len(catalog)} products after cleaning")

    rec = ContentBasedRecommender(catalog)
    print(f"TF-IDF vocabulary size: {len(rec.vectorizer.get_feature_names_out())}")

    # a couple of ordinary within-category examples
    sample = catalog.sample(3, random_state=42)
    for pid in sample["product_id"]:
        print_recommendations(catalog, rec, pid)

    # the polka dot mismatch - see README for why this one's interesting
    polka_top = catalog.products[
        catalog.products["product_name"].str.contains("Gypsy Soul Casual", na=False)
    ]
    if not polka_top.empty:
        print_recommendations(catalog, rec, polka_top.iloc[0]["product_id"])


if __name__ == "__main__":
    main()
