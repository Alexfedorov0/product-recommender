# Content-Based Product Recommender

Given a product, recommend similar products based only on their text descriptions.
No user data, no purchase history, no ratings - just "does this product read like
that product". This is the simplest recommender approach that exists, and it's a
useful baseline to understand before reaching for anything collaborative.

**Live demo:** [product-recommender.onrender.com](https://product-recommender.onrender.com) *(free tier - first load can take ~30s to wake up)*

## The business question

If a shopper is looking at a product page, what else should we show them in a
"similar items" block when we have no browsing or purchase history for that user
(a new visitor, or a store that just launched)? Content-based filtering answers
this using only the product catalog itself - it works from day one, unlike
collaborative filtering, which needs interaction data to bootstrap.

## Data

[Flipkart E-commerce Dataset](https://www.kaggle.com/datasets/atharvjairath/flipkart-ecommerce-dataset)
from Kaggle - ~20,000 product listings crawled from Flipkart.com in 2016.

After cleaning: **13,935 products** across 35+ categories (clothing, jewellery,
footwear, electronics, home goods, automotive parts, and more).

Cleaning removed ~6,000 rows, mostly ones where the description was pure
marketing boilerplate ("Buy X only for Rs. Y from Flipkart.com...") with almost
nothing product-specific left over once that's stripped out - too thin to
vectorize meaningfully.

## Approach

1. Strip boilerplate sales copy and section labels from descriptions (regex-based -
   see `product_catalog.py`). This step matters more than it sounds: left in, generic
   phrases like "genuine products" and "cash on delivery" appear in nearly every
   listing and would swamp the actual product signal.
2. Vectorize descriptions with TF-IDF (unigrams + bigrams, plus a custom stopword
   list on top of the standard English one, since e-commerce boilerplate words
   like "buy" and "flipkart" aren't in any standard stopword list).
3. For a query product, rank every other product by cosine similarity to its
   TF-IDF vector and return the top N.

Similarity is computed against one row at a time rather than building a full
NxN similarity matrix - at this catalog size that matrix would be roughly 1.5GB
of floats for no benefit, since only one row is ever needed per recommendation.

## Results

**Works as expected for near-duplicate and closely related listings:**

```
QUERY [Clothing]: Oviyon Printed Men's Round Neck T-Shirt
  0.925  Oviyon Printed Men's Round Neck T-Shirt (color variant)
  0.890  Oviyon Printed Men's Round Neck T-Shirt (color variant)
  0.839  Oviyon Printed Men's V-neck T-Shirt
```

**Reasonable generalization within a category:**

```
QUERY [Clothing]: Govind Chikan Formal Solid Women's Kurti
  0.700  Cruzaar Formal Solid Women's Kurti
  0.456  Shop Rajasthan Casual, Formal Solid, Printed Women's Kurti
  0.451  Flora Solid Women's Kurti
```

**A genuine limitation, not a cherry-picked failure:**

```
QUERY [Clothing]: Gypsy Soul Casual Short Sleeve Polka Print Women's Top
  0.311  [Tools & Hardware] Super Drool Polka Dot Plant Container Set
```

A women's top gets matched to a plant pot. Checking which terms drove this
(`recommender.explain()`) shows both listings score highest on "polka dot" and
"dot" - TF-IDF has no idea that "polka dot" describes a fabric pattern in one
listing and literally nothing about the object in the other. It just sees two
documents that share a distinctive two-word phrase. This is the core limitation
of bag-of-words methods: they match vocabulary, not meaning. An embedding-based
approach (e.g. sentence-transformers) would likely handle this correctly, since
it captures semantic context rather than surface tokens - a natural next step
if this were pushed further.

## Limitations

- **No personalization.** Every user gets the same recommendations for a given
  product - there's no notion of user preference at all.
- **Cold start works, but so does the opposite problem**: a product with a thin
  or generic description gets weak recommendations no matter how good the
  underlying model is - the method is only as good as the text it's given.
- **Bag-of-words has no semantics**, as shown above. Synonyms, misspellings, and
  coincidental phrase overlap are all treated the same way.
- **Categories aren't used to correct the model** - they're only used here to
  sanity-check results, not fed into the similarity calculation itself. Blending
  in category as a feature would likely fix cases like the polka dot mismatch.

## Running it

```bash
pip install -r requirements.txt
python demo.py                  # prints example recommendations in the terminal
pytest test_recommender.py -v   # test suite
uvicorn app:app --reload        # web demo at http://localhost:8000
```

Requires `flipkart_com-ecommerce_sample.csv` (from the Kaggle link above) in
the project root.

## Web app

`app.py` wraps the recommender in a small FastAPI app: search a product,
open its page, see the top-8 recommendations ranked by similarity. The
catalog is loaded and the TF-IDF model fit once at startup (a few seconds),
not per request.

- `/` - search / browse
- `/product/{product_id}` - a product and its recommendations
- `/api/recommend/{product_id}?top_n=5` - same thing as JSON

Deployed on Render's free tier - build command `pip install -r requirements.txt`,
start command `uvicorn app:app --host 0.0.0.0 --port $PORT` (see `Procfile`).
