"""Loads and cleans the Flipkart product catalog used by the recommender.

Data source: Flipkart E-commerce Dataset (Kaggle, ~20k products crawled
in 2016). https://www.kaggle.com/datasets/atharvjairath/flipkart-ecommerce-dataset
"""

import re
import pandas as pd


# Flipkart descriptions are full of boilerplate sales copy that has nothing
# to do with the actual product ("Buy X only for Rs. Y from Flipkart.com.
# Only Genuine Products. 30 Day Replacement Guarantee..."). Left in, this
# text dominates the TF-IDF vectors with generic words instead of anything
# that distinguishes one product from another, so it gets stripped first.
BOILERPLATE_PATTERNS = [
    r"Flipkart\.com:?\s*",
    r"Flipkart has\b",
    r"Buy .*? only for Rs\.?\s*[\d,]+\.?\d*\s*(from Flipkart\.com)?\.?",
    r"Buy .*? for Rs\.?\s*[\d,]+\.?\d*\s*online\.?",
    r"Price:?\s*Rs\.?\s*[\d,]+\.?\d*",
    r"at best prices?( with)?\b",
    r"Only Genuine Products\.?",
    r"\d+\s*Day Replacement Guarantee\.?",
    r"Free Shipping\.?",
    r"Cash On Delivery!?",
    r"&amp;|&nbsp;|&\s*\.",
]
BOILERPLATE_RE = re.compile("|".join(BOILERPLATE_PATTERNS), flags=re.IGNORECASE)

# "Key Features of X" / "Specifications of X" are section labels, not
# content. The product name inside them is redundant with product_name.
SECTION_LABEL_RE = re.compile(
    r"(Key Features of|Specifications of)\s+.*?(?=\s[A-Z]|$)", flags=re.IGNORECASE
)

WHITESPACE_RE = re.compile(r"\s+")


def clean_description(text):
    if not isinstance(text, str):
        return ""
    text = BOILERPLATE_RE.sub(" ", text)
    text = SECTION_LABEL_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def extract_top_category(category_tree):
    """product_category_tree looks like '["Clothing >> Women's ... "]'.

    A small number of rows have no real hierarchy in there at all - just
    the product name again, with no '>>' separator - which would otherwise
    get parsed as if it were the category. Those get labeled Unknown
    instead of showing garbled product text as a "category".
    """
    if not isinstance(category_tree, str):
        return "Unknown"
    match = re.search(r'"\[?"?([^">]+)', category_tree)
    if not match:
        return "Unknown"
    top = match.group(1).strip()
    if ">>" not in category_tree:
        return "Unknown"
    return top


class ProductCatalog:
    """Holds the cleaned product catalog as a DataFrame."""

    def __init__(self, csv_path, min_description_length=40):
        self.csv_path = csv_path
        self.min_description_length = min_description_length
        self.products = self._load()

    def _load(self):
        df = pd.read_csv(self.csv_path)

        df = df.dropna(subset=["product_name", "description"]).copy()
        df["description_clean"] = df["description"].apply(clean_description)
        df["category"] = df["product_category_tree"].apply(extract_top_category)

        # A handful of rows end up with almost nothing left after cleaning
        # (descriptions that were pure boilerplate) - drop those rather than
        # feeding near-empty text into the vectorizer.
        df = df[df["description_clean"].str.len() >= self.min_description_length]

        df = df.drop_duplicates(subset=["product_name", "description_clean"])
        df = df.reset_index(drop=True)
        df["product_id"] = df.index

        return df[[
            "product_id", "product_name", "category",
            "description_clean", "brand", "retail_price",
        ]].rename(columns={"description_clean": "description"})

    def get_product(self, product_id):
        row = self.products[self.products["product_id"] == product_id]
        if row.empty:
            raise KeyError(f"No product with id {product_id}")
        return row.iloc[0]

    def sample(self, n=5, category=None, random_state=None):
        pool = self.products
        if category:
            pool = pool[pool["category"] == category]
        return pool.sample(n=min(n, len(pool)), random_state=random_state)

    def __len__(self):
        return len(self.products)
