

import os, re
import pandas as pd
import numpy as np

# Data upload and consolidation
DATA_DIR   = "data"
TSV_FILES  = {
    "Apparel": os.path.join(DATA_DIR, "amazon_reviews_us_Apparel_v1_00.tsv"),
    "Beauty":  os.path.join(DATA_DIR, "amazon_reviews_us_Beauty_v1_00.tsv"),
}
CLEAN_PATH = os.path.join(DATA_DIR, "reviews_clean.csv")

# Rows to read per file (both files are ~3M+ rows 60k each is plenty)
ROWS_PER_FILE = 60_000

# Building Return reason taxonomy 
# Keywords tuned for Apparel + Beauty products
RETURN_REASONS = {
    "Size / Fit Issue": [
        "too small", "too big", "runs small", "runs large", "wrong size",
        "sizing", "too tight", "too loose", "too short", "too long",
        "didn't fit", "does not fit", "size chart", "fit perfectly wrong",
        "too narrow", "too wide", "size issue",
    ],
    "Quality Issue": [
        "poor quality", "bad quality", "cheap", "flimsy", "fell apart",
        "broke", "material", "fabric", "stitching", "peeling", "fading",
        "not durable", "thin", "fragile", "poorly made", "cheap plastic",
        "cheap material", "feels cheap", "low quality", "bad fabric",
    ],
    "Wrong / Missing Item": [
        "wrong item", "wrong product", "not what i ordered", "not as pictured",
        "different product", "missing", "incomplete", "wrong color",
        "wrong shade", "wrong size sent", "sent wrong", "received wrong",
        "different color", "not the right",
    ],
    "Damaged / Defective": [
        "damaged", "defective", "cracked", "scratched", "broken",
        "not working", "doesn't work", "stopped working", "dead on arrival",
        "doa", "faulty", "arrived broken", "arrived damaged", "leaked",
        "spilled", "shattered",
    ],
    "Expectation Mismatch": [
        "not as described", "misleading", "false advertising",
        "looks nothing like", "not worth", "waste of money",
        "overhyped", "fake", "counterfeit", "disappointed",
        "not genuine", "description is wrong", "photo is misleading",
        "looks different", "not what i expected", "not as shown",
    ],
    "Delivery / Packaging": [
        "damaged in transit", "poorly packed", "shipping damage",
        "late delivery", "never arrived", "lost in transit",
        "wrong address", "crushed box", "no bubble wrap", "bad packaging",
        "package was damaged", "arrived late",
    ],
}

NO_RETURN_LABEL = "No Return Risk"

# Data cleaning
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)           # strip HTML
    text = re.sub(r"[^a-z0-9\s]", " ", text)        # keep alphanumeric
    text = re.sub(r"\s+", " ", text).strip()
    return text


def label_return_reason(text: str) -> str:
    t = text.lower()
    for reason, keywords in RETURN_REASONS.items():
        if any(kw in t for kw in keywords):
            return reason
    return NO_RETURN_LABEL


def load_tsv(path: str, category: str, nrows: int) -> pd.DataFrame:
    print(f"Loading {category}: {os.path.basename(path)}  ({nrows:,} rows)...")
    df = pd.read_csv(
        path,
        sep="\t",
        nrows=nrows,
        on_bad_lines="skip",     
        low_memory=False,
        dtype=str,       # read everything as str first to avoid dtype warnings
    )
    df["source_category"] = category
    print(f"{len(df):,} rows loaded, {df.shape[1]} columns")
    return df


def check_files():
    missing = [path for path in TSV_FILES.values() if not os.path.exists(path)]
    if missing:
        print("\n" + "═" * 62)
        print("Missing dataset file(s):")
        for p in missing:
            print(f"      {p}")
        print("\n  Steps to fix:")
        print("  1. Go to: https://www.kaggle.com/datasets/cynthiarempel/")
        print("             amazon-us-customer-reviews-dataset")
        print("  2. Download:")
        print("       amazon_reviews_us_Apparel_v1_00.tsv")
        print("       amazon_reviews_us_Beauty_v1_00.tsv")
        print(f"  3. Place both files inside:  {DATA_DIR}/")
        print("═" * 62 + "\n")
        raise FileNotFoundError(f"Missing: {missing}")


"""Load and merge both TSV files into one DataFrame."""
def merge_datasets() -> pd.DataFrame:
    
    check_files()
    dfs = []
    for category, path in TSV_FILES.items():
        df = load_tsv(path, category, ROWS_PER_FILE)
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)
    print(f"\nMerged: {len(merged):,} total rows")
    return merged


"""Rename TSV columns to internal schema."""
def normalise(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.rename(columns={
        "review_body":     "review_text",
        "review_headline": "review_summary",
        "star_rating":     "rating",
        "product_title":   "product_name",
    })
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["helpful_votes"] = pd.to_numeric(df.get("helpful_votes", pd.Series()), errors="coerce")
    return df

"""Combine headline + body, clean, and label."""
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    
    summary = df.get("review_summary", pd.Series([""] * len(df))).fillna("")
    body    = df.get("review_text",    pd.Series([""] * len(df))).fillna("")
    df["full_review"]       = (summary + " " + body).str.strip()
    df["full_review_clean"] = df["full_review"].apply(clean_text)
    df = df[df["full_review_clean"].str.len() > 15].reset_index(drop=True)
    df["return_reason"]     = df["full_review_clean"].apply(label_return_reason)
    return df


def print_stats(df: pd.DataFrame):
    n = len(df)
    print(f"\n{'─'*62}")
    print(f"Return reason distribution  ({n:,} total reviews)")
    print(f"{'─'*62}")
    counts = df["return_reason"].value_counts()
    for reason, cnt in counts.items():
        bar = "█" * int(cnt / counts.max() * 28)
        print(f"  {reason:<30} {cnt:>6,}  ({cnt/n*100:4.1f}%)  {bar}")

    print(f"\n{'─'*62}")
    print("Reviews per category")
    print(f"{'─'*62}")
    for cat, cnt in df["source_category"].value_counts().items():
        print(f"  {cat:<15} {cnt:>7,}")

    print(f"\nAvg star rating : {df['rating'].mean():.2f}")
    print(f"Unique products : {df['product_id'].nunique():,}")
    print(f"{'─'*62}\n")


def run():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("\n" + "═"*62)
    print("  preprocess.py")
    print("  Merging Apparel + Beauty datasets")
    print("═"*62 + "\n")

    df = merge_datasets()
    df = normalise(df)
    df = build_features(df)
    print_stats(df)

    # Keep only columns needed downstream
    keep = [c for c in [
        "product_id", "product_name", "product_category", "source_category",
        "review_summary", "review_text", "full_review", "full_review_clean",
        "rating", "helpful_votes", "verified_purchase", "review_date",
        "return_reason",
    ] if c in df.columns]

    df[keep].to_csv(CLEAN_PATH, index=False)
    print(f" Saved → {CLEAN_PATH}")
    print(f"Rows: {len(df):,}  |  Columns: {len(keep)}")
    print(f"Columns: {keep}\n")
    return df[keep]


if __name__ == "__main__":
    run()
