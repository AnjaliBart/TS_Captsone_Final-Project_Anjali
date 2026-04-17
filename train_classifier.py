"""

Trains TF-IDF + Logistic Regression on reviews_clean.csv

"""

import os, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

CLEAN_PATH   = "data/reviews_clean.csv"
MODEL_PATH   = "model/return_classifier.pkl"
REPORT_PATH  = "model/classification_report.txt"
CM_PATH      = "model/confusion_matrix.png"

EXCLUDE_LABEL = "No Return Risk"
MIN_SAMPLES   = 50    # drop classes with fewer samples than this

BG    = "#0F1117"
CARD  = "#1E2130"
FG    = "#F1F5F9"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": CARD,
    "axes.edgecolor": "#2D3748", "axes.labelcolor": FG,
    "xtick.color": FG, "ytick.color": FG, "text.color": FG,
})


def load_data():
    df = pd.read_csv(CLEAN_PATH)
    df = df[df["return_reason"] != EXCLUDE_LABEL].copy()
    counts = df["return_reason"].value_counts()
    valid  = counts[counts >= MIN_SAMPLES].index.tolist()
    df     = df[df["return_reason"].isin(valid)].reset_index(drop=True)
    print(f"Training data: {len(df):,} rows, {df['return_reason'].nunique()} classes")
    print(df["return_reason"].value_counts().to_string())
    return df


def build_pipeline(class_weights):
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=30_000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=3,
            strip_accents="unicode",
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=1.0,
            class_weight=class_weights,
            solver="lbfgs",
            multi_class="multinomial",
            random_state=42,
        )),
    ])


def plot_confusion_matrix(y_test, y_pred, labels):
    cm      = confusion_matrix(y_test, y_pred, labels=labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm_norm, annot=True, fmt=".2f",
                xticklabels=labels, yticklabels=labels,
                cmap="Blues", ax=ax, linewidths=0.5, linecolor=CARD)
    ax.set_title("Return Reason Classifier — Normalised Confusion Matrix",
                 fontsize=13, pad=12, color=FG)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual",    fontsize=11)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(CM_PATH, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Confusion matrix → {CM_PATH}")


def run():
    os.makedirs("model", exist_ok=True)

    print(f"\n{'='*55}\n train_classifier.py\n{'='*55}\n")

    df     = load_data()
    X      = df["full_review_clean"].tolist()
    y      = df["return_reason"].tolist()
    labels = sorted(df["return_reason"].unique().tolist())

    # Balanced class weights
    cw_vals       = compute_class_weight("balanced", classes=np.array(labels), y=np.array(y))
    class_weights = dict(zip(labels, cw_vals))
    print(f"\n Class weights: {class_weights}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {len(X_train):,}  |  Test: {len(X_test):,}")

    # Train
    print("\nTraining TF-IDF + Logistic Regression...")
    pipeline = build_pipeline(class_weights)
    pipeline.fit(X_train, y_train)

    # Cross-validate
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5,
                                scoring="f1_macro", n_jobs=-1)
    print(f"5-Fold CV F1 (macro): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Test evaluation
    y_pred = pipeline.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=labels)

    print(f"\n Test Accuracy: {acc:.4f}")
    print(f"\n Classification Report:\n\n{report}")

    with open(REPORT_PATH, "w") as f:
        f.write(f"Test Accuracy: {acc:.4f}\n\nDataset: Apparel + Beauty (Amazon US)\n\n")
        f.write(report)
    print(f" Report → {REPORT_PATH}")

    # Confusion matrix
    plot_confusion_matrix(y_test, y_pred, labels)

    # Save model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"pipeline": pipeline, "labels": labels}, f)
    print(f" Model  → {MODEL_PATH}")
    print(f"\n Done! Accuracy: {acc:.2%}\n")


if __name__ == "__main__":
    run()
