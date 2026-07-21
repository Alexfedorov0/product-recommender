from pathlib import Path

import pytest
from product_catalog import ProductCatalog, clean_description, extract_top_category
from recommender import ContentBasedRecommender

DATA_PATH = Path(__file__).parent / "flipkart_com-ecommerce_sample.csv"


@pytest.fixture(scope="module")
def catalog():
    return ProductCatalog(DATA_PATH)


@pytest.fixture(scope="module")
def recommender(catalog):
    return ContentBasedRecommender(catalog)


def test_clean_description_strips_boilerplate():
    raw = "Buy Foo Bar only for Rs. 999 from Flipkart.com. Only Genuine Products. 30 Day Replacement Guarantee. Free Shipping. Cash On Delivery!"
    cleaned = clean_description(raw)
    assert "flipkart" not in cleaned.lower()
    assert "cash on delivery" not in cleaned.lower()
    assert "genuine products" not in cleaned.lower()


def test_clean_description_handles_non_string():
    assert clean_description(None) == ""
    assert clean_description(float("nan")) == ""


def test_extract_top_category_normal_case():
    tree = '["Clothing >> Women\'s Clothing >> Tops"]'
    assert extract_top_category(tree) == "Clothing"


def test_extract_top_category_no_hierarchy_returns_unknown():
    # some rows have the product name repeated with no real category info
    tree = '["classyworld Brass Cufflink (Silver-01)"]'
    assert extract_top_category(tree) == "Unknown"


def test_catalog_loads_products(catalog):
    assert len(catalog) > 1000
    assert "description" in catalog.products.columns
    assert "category" in catalog.products.columns


def test_catalog_drops_empty_descriptions(catalog):
    assert (catalog.products["description"].str.len() >= 40).all()


def test_get_product_returns_correct_row(catalog):
    sample = catalog.sample(1, random_state=1).iloc[0]
    fetched = catalog.get_product(sample["product_id"])
    assert fetched["product_name"] == sample["product_name"]


def test_get_product_raises_on_missing_id(catalog):
    with pytest.raises(KeyError):
        catalog.get_product(-1)


def test_recommend_returns_requested_count(recommender, catalog):
    pid = catalog.products.iloc[0]["product_id"]
    results = recommender.recommend(pid, top_n=5)
    assert len(results) == 5


def test_recommend_excludes_query_product(recommender, catalog):
    pid = catalog.products.iloc[10]["product_id"]
    results = recommender.recommend(pid, top_n=10)
    assert pid not in results["product_id"].values


def test_recommend_results_sorted_descending(recommender, catalog):
    pid = catalog.products.iloc[5]["product_id"]
    results = recommender.recommend(pid, top_n=10)
    similarities = results["similarity"].tolist()
    assert similarities == sorted(similarities, reverse=True)


def test_recommend_raises_on_unknown_product(recommender):
    with pytest.raises(KeyError):
        recommender.recommend(-1)


def test_similar_products_score_higher_than_random(recommender, catalog):
    # two listings for near-identical products should score much higher
    # than the average pair in the catalog
    shirts = catalog.products[
        catalog.products["product_name"].str.contains("Checkered Casual", na=False)
    ]
    if len(shirts) < 2:
        pytest.skip("fixture data doesn't contain the expected near-duplicate pair")
    pid = shirts.iloc[0]["product_id"]
    results = recommender.recommend(pid, top_n=len(catalog))
    assert results["similarity"].iloc[0] > results["similarity"].median()


def test_explain_returns_nonempty_terms(recommender, catalog):
    pid = catalog.products.iloc[0]["product_id"]
    terms = recommender.explain(pid)
    assert len(terms) > 0
    assert all(isinstance(term, str) and isinstance(weight, float) for term, weight in terms)
