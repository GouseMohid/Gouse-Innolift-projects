from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import nltk
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from model import clean_reviews, load_dataset, run_eda


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
CHART_DIR = BASE_DIR / "charts"
MODEL_PATH = MODEL_DIR / "model.pkl"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"


def ensure_nltk_stopwords():
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)


def save_charts(reviews):
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 4))
    sentiment_counts = reviews["sentiment"].value_counts().reindex(
        ["Negative", "Neutral", "Positive"],
        fill_value=0,
    )
    plt.bar(sentiment_counts.index, sentiment_counts.values, color=["#d94f45", "#f0b429", "#2f9e44"])
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Review Count")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(CHART_DIR / "sentiment_distribution.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    rating_means = reviews.groupby("sentiment")["rating"].mean().reindex(
        ["Negative", "Neutral", "Positive"],
        fill_value=0,
    )
    plt.bar(rating_means.index, rating_means.values, color=["#d94f45", "#f0b429", "#2f9e44"])
    plt.title("Average Rating by Sentiment")
    plt.xlabel("Sentiment")
    plt.ylabel("Average Rating")
    plt.ylim(0, 5)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(CHART_DIR / "rating_by_sentiment.png", dpi=160)
    plt.close()

    correlation_frame = reviews.copy()
    correlation_frame["sentiment_code"] = correlation_frame["sentiment"].map(
        {"Negative": 0, "Neutral": 1, "Positive": 2}
    )
    corr = correlation_frame[["rating", "text_length", "sentiment_code"]].corr()
    plt.figure(figsize=(6, 4))
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Correlation")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=30, ha="right")
    plt.yticks(range(len(corr.index)), corr.index)
    for row_index, row_name in enumerate(corr.index):
        for column_index, column_name in enumerate(corr.columns):
            value = corr.loc[row_name, column_name]
            plt.text(column_index, row_index, f"{value:.2f}", ha="center", va="center")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "correlation_heatmap.png", dpi=160)
    plt.close()

    print(f"\nSaved chart PNGs to: {CHART_DIR}")


def train_model():
    ensure_nltk_stopwords()
    raw_reviews = load_dataset()
    run_eda(raw_reviews)
    reviews = clean_reviews(raw_reviews)

    print("\nAfter cleaning:")
    print(f"Shape: {reviews.shape}")
    print("\nNull counts:")
    print(reviews.isnull().sum())
    print(f"\nVerified 0 nulls after cleaning: {reviews.isnull().sum().sum() == 0}")
    print("\nSentiment distribution:")
    print(reviews["sentiment"].value_counts())
    print("\nAverage rating by sentiment:")
    print(reviews.groupby("sentiment")["rating"].mean())

    save_charts(reviews)

    vectorizer = TfidfVectorizer(max_features=6000, ngram_range=(1, 2))
    features = vectorizer.fit_transform(reviews["cleaned_review"])
    labels = reviews["sentiment"]

    min_per_class = labels.value_counts().min()
    stratify = labels if min_per_class >= 2 else None
    n_classes = labels.nunique()
    test_size = max(0.2, n_classes / len(labels))
    if stratify is not None:
        test_size = max(test_size, n_classes / max(1, (len(labels) - 1)))

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=42,
        stratify=stratify,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"\nAccuracy: {accuracy:.2%}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as model_file:
        pickle.dump(model, model_file)
    with VECTORIZER_PATH.open("wb") as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)

    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved vectorizer to: {VECTORIZER_PATH}")


if __name__ == "__main__":
    train_model()
