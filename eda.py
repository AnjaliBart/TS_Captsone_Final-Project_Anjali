"""
Exploratory Data Analysis and visualization

"""

import os, re, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud

warnings.filterwarnings("ignore")

CLEAN_PATH = "data/reviews_clean.csv"
OUTPUT_DIR = "eda_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PALETTE = {
    "Size / Fit Issue":       "#4C9BE8",
    "Quality Issue":          "#E8734C",
    "Wrong / Missing Item":   "#8B5CF6",
    "Damaged / Defective":    "#EC4899",
    "Expectation Mismatch":   "#EF4444",
    "Delivery / Packaging":   "#10B981",
    "No Return Risk":         "#94A3B8",
}
ACCENT  = "#EF4444"
BG      = "#0F1117"
CARD    = "#1E2130"
FG      = "#F1F5F9"

plt.rcParams.update({
    "figure.facecolor": BG,   "axes.facecolor":  CARD,
    "axes.edgecolor":   "#2D3748", "axes.labelcolor": FG,
    "xtick.color": FG,  "ytick.color": FG,  "text.color": FG,
    "grid.color":  "#2D3748", "grid.linestyle": "--", "grid.alpha": 0.5,
    "axes.titlesize": 13, "axes.titlepad": 12, "axes.titleweight": "bold",
})

def save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f" {name}")

STOPWORDS = {
    "the","and","to","a","of","is","in","it","this","was","for","i","have",
    "my","with","that","not","but","be","on","so","an","at","are","had","we",
    "he","she","they","our","your","its","product","item","amazon","order",
    "bought","get","got","one","would","will","very","much","from","by","or",
    "if","no","do","as","just","also","about","more","when","out","up","what",
    "how","can","did","they","use","used","using","great","good","love","like",
    "really","only","even","color","colour","me","so","has","them","these",
}

NEGATIVE_SIGNALS = [
    "not as described","misleading","poor quality","cheap","broke","wrong size",
    "doesn't fit","waste of money","disappointed","return","refund","fake",
    "not worth","bad quality","falls apart","not what i expected",
    "false advertising","looks nothing like",
]

def neg_count(text):
    return sum(1 for s in NEGATIVE_SIGNALS if s in str(text).lower())

def get_bigrams(texts, n=10):
    bg = []
    for t in texts:
        w = str(t).split()
        bg.extend(zip(w[:-1], w[1:]))
    return Counter(bg).most_common(n)


#  1. Return Reason Distribution 
def plot_return_reason(df):
    print("[1/10] Return reason distribution")
    counts = df["return_reason"].value_counts()
    colors = [PALETTE.get(r,"#94A3B8") for r in counts.index]
    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.barh(counts.index, counts.values, color=colors, height=0.6, edgecolor="none")
    for bar, val in zip(bars, counts.values):
        ax.text(val + counts.max()*0.01, bar.get_y()+bar.get_height()/2,
                f"{val:,}  ({val/len(df)*100:.1f}%)", va="center", fontsize=9.5, color=FG)
    ax.set_title("Return Reason Distribution(Apparel + Beauty)")
    ax.set_xlabel("Number of Reviews")
    ax.invert_yaxis(); ax.grid(axis="x")
    ax.set_xlim(0, counts.max()*1.22)
    fig.tight_layout(); save(fig, "01_return_reason_distribution.png")


# 2. Category Breakdown 
def plot_category_breakdown(df):
    print("[2/10] Category breakdown (Apparel vs Beauty)")
    cats = df["source_category"].value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: stacked bar for return reasons per category
    cat_reason = df.groupby(["source_category", "return_reason"]).size().unstack(fill_value=0)
    cat_reason_pct = cat_reason.div(cat_reason.sum(axis=1), axis=0) * 100
    bottom = np.zeros(len(cat_reason_pct))
    for reason in cat_reason_pct.columns:
        color = PALETTE.get(reason, "#94A3B8")
        axes[0].bar(cat_reason_pct.index, cat_reason_pct[reason],
                    bottom=bottom, color=color, label=reason, edgecolor="none")
        bottom += cat_reason_pct[reason].values
    axes[0].set_title("Return Reason Mix per Category (%)")
    axes[0].set_ylabel("% of Reviews")
    axes[0].legend(fontsize=7.5, loc="upper right",
                   facecolor=CARD, edgecolor="#2D3748", labelcolor=FG)

    # Right: pie for category split
    axes[1].pie(cats.values, labels=cats.index,
                colors=["#4C9BE8", "#EC4899"],
                autopct="%1.1f%%", startangle=90,
                wedgeprops={"edgecolor": BG, "linewidth": 2},
                textprops={"color": FG, "fontsize": 11})
    axes[1].set_title("Dataset Split by Category")

    fig.suptitle("Apparel vs Beauty Category Overview", fontsize=14, color=FG, y=1.01)
    fig.tight_layout(); save(fig, "02_category_breakdown.png")


#  3. Star Rating Distribution 
def plot_rating_distribution(df):
    if "rating" not in df.columns:
        print(" No rating column"); return
    print("[3/10] Star rating distribution")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, cat in zip(axes, ["Apparel", "Beauty"]):
        sub = df[df["source_category"] == cat]["rating"].dropna()
        counts = sub.value_counts().sort_index()
        colors = ["#EF4444","#F97316","#EAB308","#84CC16","#22C55E"]
        ax.bar(counts.index.astype(int), counts.values,
               color=colors[:len(counts)], edgecolor="none", width=0.6)
        ax.set_title(f"{cat}  Star Rating Distribution")
        ax.set_xlabel("Stars"); ax.set_ylabel("Count")
        ax.set_xticks([1,2,3,4,5]); ax.grid(axis="y")
        for x, y in zip(counts.index.astype(int), counts.values):
            ax.text(x, y + counts.max()*0.01, f"{y:,}", ha="center", fontsize=8.5, color=FG)
    fig.tight_layout(); save(fig, "03_star_rating_distribution.png")


#  4. Review Length Distribution 
def plot_review_length(df):
    print("[4/10] Review length distribution")
    df = df.copy()
    df["review_len"] = df["full_review_clean"].str.split().str.len()
    risky   = df[df["return_reason"] != "No Return Risk"]["review_len"]
    no_risk = df[df["return_reason"] == "No Return Risk"]["review_len"]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    bins = range(0, 250, 8)
    ax.hist(risky.clip(0,250),   bins=bins, alpha=0.75, color=ACCENT,    label="Return-Risk Reviews", density=True)
    ax.hist(no_risk.clip(0,250), bins=bins, alpha=0.55, color="#4C9BE8", label="No Return Risk",       density=True)
    ax.set_xlabel("Word Count"); ax.set_ylabel("Density")
    ax.set_title("Review Length: Return Risk vs No Risk")
    ax.legend(facecolor=CARD, edgecolor="#2D3748")
    ax.grid(axis="y"); fig.tight_layout(); save(fig, "04_review_length_distribution.png")


#  5. Word Cloud  Risk Reviews 
def plot_wordcloud_risk(df):
    print("[5/10] Word cloud return risk reviews")
    text = " ".join(df[df["return_reason"] != "No Return Risk"]["full_review_clean"].dropna())
    wc = WordCloud(width=1000, height=420, background_color=BG,
                   colormap="RdYlGn_r", max_words=120, stopwords=STOPWORDS,
                   collocations=False, prefer_horizontal=0.85).generate(text)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
    ax.set_title("Most Frequent Words in Return-Risk Reviews", color=FG, fontsize=14, pad=12)
    fig.tight_layout(); save(fig, "05_wordcloud_risk.png")


#  6. Word Cloud for Apparel vs Beauty 
def plot_wordcloud_categories(df):
    print("[6/10] Word cloud Apparel vs Beauty")
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, cat, cmap in [
        (axes[0], "Apparel", "Blues"),
        (axes[1], "Beauty",  "RdPu"),
    ]:
        text = " ".join(df[df["source_category"]==cat]["full_review_clean"].dropna())
        wc = WordCloud(width=700, height=370, background_color=BG,
                       colormap=cmap, max_words=90, stopwords=STOPWORDS,
                       collocations=False).generate(text)
        ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
        ax.set_title(f"{cat} Reviews Key Words", fontsize=13, color=FG, pad=10)
    fig.suptitle("Vocabulary Comparison: Apparel vs Beauty", fontsize=15, color=FG, y=1.02)
    fig.tight_layout(); save(fig, "06_wordcloud_apparel_vs_beauty.png")


#  7. Top Bigrams per Return Reason 
def plot_bigrams(df):
    print("[7/10] Top bigrams per return reason")
    reasons = [r for r in df["return_reason"].value_counts().index if r != "No Return Risk"][:4]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, reason in zip(axes.flatten(), reasons):
        texts   = df[df["return_reason"]==reason]["full_review_clean"].dropna().tolist()
        bigrams = get_bigrams(texts, 8)
        labels  = [f"{a} {b}" for (a,b),_ in bigrams]
        counts  = [c for _,c in bigrams]
        ax.barh(labels[::-1], counts[::-1],
                color=PALETTE.get(reason, ACCENT), alpha=0.85, edgecolor="none")
        ax.set_title(reason, color=FG, fontsize=11)
        ax.grid(axis="x"); ax.tick_params(labelsize=9)
    fig.suptitle("Top Bigrams by Return Reason", fontsize=15, color=FG, y=1.01)
    fig.tight_layout(); save(fig, "07_bigrams_per_reason.png")


#  8. Heatmap for Return Reason × Star Rating 
def plot_heatmap(df):
    if "rating" not in df.columns:
        print(" No rating:skipping heatmap"); return
    print("[8/10] Return reason x star rating heatmap")
    sub = df[df["return_reason"] != "No Return Risk"].copy()
    sub["rating_bin"] = sub["rating"].fillna(0).astype(int).clip(1, 5)
    pivot = sub.groupby(["return_reason","rating_bin"]).size().unstack(fill_value=0)
    pivot = pivot.div(pivot.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd",
                linewidths=0.5, linecolor=CARD,
                cbar_kws={"label": "% of category"}, ax=ax)
    ax.set_title("Return Reason vs Star Rating  (% within each reason)")
    ax.set_xlabel("Star Rating"); ax.set_ylabel("")
    ax.tick_params(labelsize=9)
    fig.tight_layout(); save(fig, "08_reason_rating_heatmap.png")


#  9. Negative Sentiment Density 
def plot_sentiment_density(df):
    print("[9/10] Negative sentiment signal density")
    df = df.copy()
    df["neg"] = df["full_review_clean"].apply(neg_count)
    means  = df.groupby("return_reason")["neg"].mean().sort_values(ascending=False)
    colors = [PALETTE.get(r, "#94A3B8") for r in means.index]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    bars = ax.bar(range(len(means)), means.values, color=colors, edgecolor="none")
    ax.set_xticks(range(len(means)))
    ax.set_xticklabels([r.replace(" / ","/\n") for r in means.index],
                        rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Avg Negative Keyword Hits per Review")
    ax.set_title("Negative Sentiment Density by Return Reason")
    ax.grid(axis="y")
    for bar, val in zip(bars, means.values):
        ax.text(bar.get_x()+bar.get_width()/2, val+0.003,
                f"{val:.2f}", ha="center", fontsize=9, color=FG)
    fig.tight_layout(); save(fig, "09_sentiment_density.png")


#  10. Summary Dashboard 
def plot_summary_dashboard(df):
    print("[10/10] Summary dashboard")
    df = df.copy()
    df["review_len"] = df["full_review_clean"].str.split().str.len()
    df["neg"]        = df["full_review_clean"].apply(neg_count)
    risk    = df[df["return_reason"] != "No Return Risk"]
    no_risk = df[df["return_reason"] == "No Return Risk"]

    fig = plt.figure(figsize=(15, 9))
    fig.suptitle("EDA Summary  (Apparel + Beauty)", fontsize=17, color=FG, y=0.99)
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.38)

    # Top-left: return reason bar
    ax1 = fig.add_subplot(gs[0, :2])
    counts = df["return_reason"].value_counts()
    colors = [PALETTE.get(r,"#94A3B8") for r in counts.index]
    ax1.barh(counts.index, counts.values, color=colors, height=0.6, edgecolor="none")
    ax1.invert_yaxis(); ax1.grid(axis="x")
    ax1.set_title("Return Reason Distribution")

    # Top-right: risk vs no-risk pie
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.pie([len(risk), len(no_risk)],
            labels=["Return Risk","No Risk"],
            colors=[ACCENT,"#10B981"],
            autopct="%1.0f%%", startangle=90,
            wedgeprops={"edgecolor": BG, "linewidth":2},
            textprops={"color":FG,"fontsize":9})
    ax2.set_title("Risk Split")

    # Bottom-left: review length boxplot
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.boxplot([risk["review_len"].clip(0,250), no_risk["review_len"].clip(0,250)],
                labels=["Risk","No Risk"], patch_artist=True,
                boxprops=dict(facecolor=ACCENT, alpha=0.6),
                medianprops=dict(color="white", linewidth=2),
                whiskerprops=dict(color=FG), capprops=dict(color=FG),
                flierprops=dict(markerfacecolor=ACCENT, markersize=3, alpha=0.3))
    ax3.set_title("Review Length (words)"); ax3.grid(axis="y")

    # Bottom-middle: avg rating by reason
    ax4 = fig.add_subplot(gs[1, 1])
    if "rating" in df.columns:
        avg = df.groupby("return_reason")["rating"].mean().sort_values()
        ax4.barh(avg.index, avg.values,
                 color=[PALETTE.get(r,"#94A3B8") for r in avg.index],
                 height=0.6, edgecolor="none")
        ax4.set_title("Avg Star Rating by Reason")
        ax4.set_xlim(0, 5.5); ax4.grid(axis="x")
        ax4.axvline(avg.mean(), color="white", linestyle="--", alpha=0.5, linewidth=1)

    # Bottom-right: neg keyword density
    ax5 = fig.add_subplot(gs[1, 2])
    neg = df.groupby("return_reason")["neg"].mean().sort_values(ascending=False)
    ax5.bar(range(len(neg)), neg.values,
            color=[PALETTE.get(r,"#94A3B8") for r in neg.index], edgecolor="none")
    ax5.set_xticks(range(len(neg)))
    ax5.set_xticklabels([r.split("/")[0].strip()[:10] for r in neg.index],
                         rotation=35, ha="right", fontsize=7.5)
    ax5.set_title("Neg. Keyword Density"); ax5.grid(axis="y")

    save(fig, "10_summary_dashboard.png")


#  Main 
def run():
    print(f"\n{'='*55}\nEDA\n{'='*55}\n")
    df = pd.read_csv(CLEAN_PATH)
    print(f"  Loaded {len(df):,} rows from {CLEAN_PATH}\n")

    plot_return_reason(df)
    plot_category_breakdown(df)
    plot_rating_distribution(df)
    plot_review_length(df)
    plot_wordcloud_risk(df)
    plot_wordcloud_categories(df)
    plot_bigrams(df)
    plot_heatmap(df)
    plot_sentiment_density(df)
    plot_summary_dashboard(df)

    print(f"\n{'='*55}")
    print(f"10 charts saved to → {OUTPUT_DIR}/")
    print(f"{'='*55}\n")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"  {f}")

if __name__ == "__main__":
    run()
