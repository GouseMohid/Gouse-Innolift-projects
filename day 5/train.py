from pathlib import Path

import joblib
import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from sentiment_model import clean_text


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
DATASET_PATH = DATA_DIR / "amazon_reviews.csv"
SAMPLE_DATASET_PATH = DATA_DIR / "amazon_reviews_sample.csv"
MODEL_PATH = MODEL_DIR / "sentiment_model.pkl"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"
MAX_REVIEWS = 10_000


def ensure_nltk_stopwords():
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)


def find_column(columns, candidates):
    normalized = {column.lower().strip(): column for column in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return None


def rating_to_sentiment(rating):
    rating = float(rating)
    if rating <= 2:
        return "Negative"
    if rating == 3:
        return "Neutral"
    return "Positive"


def load_training_data():
    dataset_path = DATASET_PATH
    if not dataset_path.exists():
        dataset_path = SAMPLE_DATASET_PATH
        print(f"Full 10K dataset not found. Using sample file: {dataset_path}")

    reviews = pd.read_csv(dataset_path).head(MAX_REVIEWS)
    text_column = find_column(
        reviews.columns,
        ["review", "review_text", "text", "content", "summary"],
    )
    rating_column = find_column(reviews.columns, ["rating", "score", "stars"])

    if text_column is None:
        raise ValueError("Dataset must include a review text column.")
    if rating_column is None:
        raise ValueError("Dataset must include a rating/score/stars column.")

    reviews = reviews[[text_column, rating_column]].rename(
        columns={text_column: "review", rating_column: "rating"}
    )
    reviews = reviews.dropna(subset=["review", "rating"]).copy()
    reviews["rating"] = pd.to_numeric(reviews["rating"], errors="coerce")
    reviews = reviews.dropna(subset=["rating"])
    reviews["cleaned_review"] = reviews["review"].apply(clean_text)
    reviews["sentiment"] = reviews["rating"].apply(rating_to_sentiment)
    reviews = reviews[reviews["cleaned_review"].str.len() > 0]

    print(f"Loaded training shape: {reviews.shape}")
    print("\nSentiment distribution:")
    print(reviews["sentiment"].value_counts())
    return reviews


def train_model():
    ensure_nltk_stopwords()
    reviews = load_training_data()

    vectorizer = TfidfVectorizer(max_features=6000, ngram_range=(1, 2))
    features = vectorizer.fit_transform(reviews["cleaned_review"])
    labels = reviews["sentiment"]

    stratify = labels if labels.value_counts().min() >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )

    model = MultinomialNB(alpha=0.5)
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"\nAccuracy: {accuracy:.2%}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved vectorizer to: {VECTORIZER_PATH}")


if __name__ == "__main__":
    train_model()
