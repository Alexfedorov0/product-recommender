"""Web demo for the content-based recommender.

Loads the catalog and fits the TF-IDF model once at startup, then serves
a small search UI plus a JSON endpoint for the recommendations themselves.
"""

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from product_catalog import ProductCatalog
from recommender import ContentBasedRecommender

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "flipkart_com-ecommerce_sample.csv"

app = FastAPI(title="Content-Based Product Recommender")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

catalog = ProductCatalog(DATA_PATH)
recommender = ContentBasedRecommender(catalog)
PRODUCT_COUNT = f"{len(catalog):,}"


def search_products(query, limit=20):
    if not query:
        return catalog.products.sample(n=min(limit, len(catalog)), random_state=None)
    mask = catalog.products["product_name"].str.contains(query, case=False, na=False)
    return catalog.products[mask].head(limit)


@app.get("/", response_class=HTMLResponse)
def index(request: Request, q: str = ""):
    results = search_products(q)
    return templates.TemplateResponse(
        request,
        "index.html",
        {"query": q, "results": results.to_dict("records"), "product_count": PRODUCT_COUNT},
    )


@app.get("/product/{product_id}", response_class=HTMLResponse)
def product_detail(request: Request, product_id: int):
    try:
        product = catalog.get_product(product_id)
    except KeyError:
        return templates.TemplateResponse(
            request,
            "not_found.html",
            {"product_id": product_id, "product_count": PRODUCT_COUNT},
            status_code=404,
        )

    recs = recommender.recommend(product_id, top_n=8).to_dict("records")
    return templates.TemplateResponse(
        request,
        "product.html",
        {"product": product.to_dict(), "recommendations": recs, "product_count": PRODUCT_COUNT},
    )


@app.get("/api/recommend/{product_id}")
def recommend_json(product_id: int, top_n: int = 5):
    try:
        results = recommender.recommend(product_id, top_n=top_n)
    except KeyError:
        return {"error": f"no product with id {product_id}"}
    return {
        "product_id": product_id,
        "recommendations": results[["product_id", "product_name", "category", "similarity"]].to_dict("records"),
    }
