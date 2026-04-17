# 🔎 E-commerce Return Risk Analyzer
### Dataset: Amazon US Reviews — Apparel + Beauty

> *"Product descriptions often overpromise. Reviews reveal the truth. The gap between the two is where returns happen."*

---

## Dataset Setup 

1. Dataset: https://www.kaggle.com/datasets/cynthiarempel/amazon-us-customer-reviews-dataset

2. Download **both** of these files:
   - `amazon_reviews_us_Apparel_v1_00.tsv`
   - `amazon_reviews_us_Beauty_v1_00.tsv`

3. Place both files in the `data/` folder:


---

## Run Order

```bash
# Install dependencies (once)
pip install -r requirements.txt

# Step 1 — Merge + clean datasets, generate return reason labels
python preprocess.py

# Step 2 — Generate 10 EDA visualizations → saved to eda_output/
python eda.py

# Step 3 — Train ML return reason classifier
python train_classifier.py

# Step 4 — Launch the Streamlit app
streamlit run app.py
```


---

## How It Works

```
Product Description + Customer Reviews
         ↓
  Module 1: Claim Extractor       → extracts promises from description
         ↓
  Module 2: Review Verifier       → Confirmed ✅ / Contradicted ❌ / Unverified ⚠️
         ↓
  Module 3: Return Reason ML      → classifies each review (6 return categories)
         ↓
  Module 4: Risk Scorer           → 0-100 return risk score
         ↓
  Streamlit Dashboard             → gauge, claim table, charts, fix suggestions
```

---


## Return Reason Classes

| Reason | Typical Triggers |
|---|---|
| Size / Fit Issue | "runs small", "too tight", "wrong size" |
| Quality Issue | "cheap fabric", "fell apart", "poor material" |
| Wrong / Missing Item | "wrong color", "not what I ordered" |
| Damaged / Defective | "arrived broken", "leaked", "defective" |
| Expectation Mismatch | "not as described", "misleading photos" |
| Delivery / Packaging | "damaged in transit", "no bubble wrap" |

---

## Deploy to Streamlit Cloud

1. Push project to GitHub (include `model/` and `data/reviews_clean.csv`)
2. Visit https://streamlit.io/cloud → New App
3. Connect repo, set `app.py` as entry point → Deploy 🎉
