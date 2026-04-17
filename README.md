# 🔎 E-commerce Return Risk Analyzer
### Dataset: Amazon US Reviews  Apparel + Beauty

E-commerce Return & Seller Blacklist Prevention System
Identifying Misleading Product Descriptions Through NLP & Machine Learning to Reduce Returns and Protect Seller Reputation


---

## Dataset Setup 

1. Dataset: https://www.kaggle.com/datasets/cynthiarempel/amazon-us-customer-reviews-dataset

2.Two files used in project
   - `amazon_reviews_us_Apparel_v1_00.tsv`
   - `amazon_reviews_us_Beauty_v1_00.tsv`


---

## Run Order

```

# Step 1 : Merge + clean datasets, generate return reason labels
python preprocess.py

# Step 2 : Generate 10 EDA visualizations
python eda.py

# Step 3 : Train ML return reason classifier
python train_classifier.py

# Step 4 : Launch the Streamlit app
streamlit run app.py
```


---

## Architchture Flow

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


