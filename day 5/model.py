from pathlib import Path

import nltk
import pandas as pd

from sentiment_model import clean_text


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_DATA_PATH = DATA_DIR / "amazon_reviews.csv"
SAMPLE_DATA_PATH = DATA_DIR / "amazon_reviews_sample.csv"
OUTPUT_PATH = DATA_DIR / "amazon_reviews_m1_cleaned.csv"
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


def sentiment_from_rating(rating):
    try:
        rating = float(rating)
    except (TypeError, ValueError):
        return "Unknown"
    if rating <= 2:
        return "Negative"
    if rating == 3:
        return "Neutral"
    return "Positive"


def load_dataset(path=None):
    dataset_path = Path(path) if path else DEFAULT_DATA_PATH
    if not dataset_path.exists():
        dataset_path = SAMPLE_DATA_PATH
        print(f"Full 10K dataset not found. Using sample file: {dataset_path}")

    reviews = pd.read_csv(dataset_path).head(MAX_REVIEWS)
    print(f"Loaded dataset: {dataset_path}")
    print(f"Shape: {reviews.shape}")
    print("\nHead:")
    print(reviews.head())
    print("\nDtypes:")
    print(reviews.dtypes)
    return reviews


def clean_reviews(frame):
    text_column = find_column(
        frame.columns,
        ["review", "review_text", "text", "content", "summary"],
    )
    rating_column = find_column(frame.columns, ["rating", "score", "stars"])

    if text_column is None:
        raise ValueError("Dataset must include a review text column.")
    if rating_column is None:
        raise ValueError("Dataset must include a rating/score/stars column.")

    cleaned = frame[[text_column, rating_column]].rename(
        columns={text_column: "review", rating_column: "rating"}
    )
    cleaned = cleaned.dropna(subset=["review", "rating"]).copy()
    cleaned["cleaned_review"] = cleaned["review"].apply(clean_text)
    cleaned["sentiment"] = cleaned["rating"].apply(sentiment_from_rating)
    cleaned = cleaned[cleaned["cleaned_review"].str.len() > 0]
    cleaned = cleaned[cleaned["sentiment"] != "Unknown"]
    return cleaned


def main():
    ensure_nltk_stopwords()
    raw_reviews = load_dataset()
    cleaned_reviews = clean_reviews(raw_reviews)

    print("\nAfter cleaning:")
    print(f"Shape: {cleaned_reviews.shape}")
    print("\nNull counts:")
    print(cleaned_reviews.isnull().sum())
    print("\nRating distribution:")
    print(cleaned_reviews["rating"].value_counts().sort_index())
    print("\nSentiment distribution:")
    print(cleaned_reviews["sentiment"].value_counts())
    print("\nCleaned preview:")
    print(cleaned_reviews[["review", "rating", "sentiment", "cleaned_review"]].head())

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned_reviews.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved M1 cleaned output to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
