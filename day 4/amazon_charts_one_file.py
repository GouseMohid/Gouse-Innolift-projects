import os
import tempfile

MATPLOTLIB_CACHE_DIR = tempfile.mkdtemp(prefix="matplotlib-")
os.environ.setdefault("MPLCONFIGDIR", MATPLOTLIB_CACHE_DIR)

import matplotlib.pyplot as plt
import pandas as pd


DATASET_PATH = "data/amazon_reviews_sample.csv"
OUTPUT_DIR = "day 4/charts"


def add_review_features(df):
    df = df.copy()
    df["review_length"] = df["review"].str.split().str.len()
    df["sentiment"] = "Neutral"
    df.loc[df["rating"] >= 4, "sentiment"] = "Positive"
    df.loc[df["rating"] <= 2, "sentiment"] = "Negative"
    return df


def save_rating_bar_chart(df):
    rating_counts = df["rating"].value_counts().sort_index()

    plt.figure(figsize=(8, 5))
    plt.bar(rating_counts.index.astype(str), rating_counts.values, color="#2f80ed")
    plt.title("Amazon Review Rating Distribution")
    plt.xlabel("Rating")
    plt.ylabel("Number of Reviews")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/rating_distribution_bar.png", dpi=150)
    plt.close()


def save_rating_length_scatter(df):
    plt.figure(figsize=(8, 5))
    plt.scatter(df["rating"], df["review_length"], color="#27ae60", alpha=0.75)
    plt.title("Review Length vs Rating")
    plt.xlabel("Rating")
    plt.ylabel("Review Length (Words)")
    plt.xticks(sorted(df["rating"].unique()))
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/rating_vs_review_length_scatter.png", dpi=150)
    plt.close()


def save_review_length_histogram(df):
    plt.figure(figsize=(8, 5))
    plt.hist(df["review_length"], bins=8, color="#f2994a", edgecolor="black")
    plt.title("Review Length Histogram")
    plt.xlabel("Review Length (Words)")
    plt.ylabel("Number of Reviews")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/review_length_histogram.png", dpi=150)
    plt.close()


def save_average_length_line_chart(df):
    average_length_by_rating = df.groupby("rating")["review_length"].mean().sort_index()

    plt.figure(figsize=(8, 5))
    plt.plot(
        average_length_by_rating.index,
        average_length_by_rating.values,
        marker="o",
        color="#9b51e0",
        linewidth=2,
    )
    plt.title("Average Review Length Trend by Rating")
    plt.xlabel("Rating")
    plt.ylabel("Average Review Length (Words)")
    plt.xticks(average_length_by_rating.index)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/average_review_length_by_rating_line.png", dpi=150)
    plt.close()


def save_custom_sentiment_rating_chart(df):
    sentiment_order = ["Negative", "Neutral", "Positive"]
    average_rating = df.groupby("sentiment")["rating"].mean().reindex(sentiment_order)
    mean_rating = df["rating"].mean()
    colors = ["#d64545", "#f2c94c", "#27ae60"]

    plt.figure(figsize=(8, 5))
    plt.bar(
        average_rating.index,
        average_rating.values,
        color=colors,
        label="Average rating by sentiment",
    )
    plt.axhline(
        mean_rating,
        color="#2f3640",
        linestyle="--",
        linewidth=2,
        label=f"Overall mean rating: {mean_rating:.2f}",
    )
    plt.title("Average Amazon Review Rating by Sentiment")
    plt.xlabel("Sentiment")
    plt.ylabel("Average Rating")
    plt.ylim(0, 5.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/custom_sentiment_average_rating_bar.png", dpi=150)
    plt.close()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    reviews = pd.read_csv(DATASET_PATH)
    reviews = add_review_features(reviews)

    save_rating_bar_chart(reviews)
    save_rating_length_scatter(reviews)
    save_review_length_histogram(reviews)
    save_average_length_line_chart(reviews)
    save_custom_sentiment_rating_chart(reviews)

    print("Saved charts:")
    print(f"- {OUTPUT_DIR}/rating_distribution_bar.png")
    print(f"- {OUTPUT_DIR}/rating_vs_review_length_scatter.png")
    print(f"- {OUTPUT_DIR}/review_length_histogram.png")
    print(f"- {OUTPUT_DIR}/average_review_length_by_rating_line.png")
    print(f"- {OUTPUT_DIR}/custom_sentiment_average_rating_bar.png")


if __name__ == "__main__":
    main()
