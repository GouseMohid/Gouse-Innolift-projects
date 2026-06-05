from pathlib import Path
import re
import string

import nltk
import pandas as pd
from nltk.corpus import stopwords


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


def clean_text(text):
    text = "" if text is None else str(text)
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    try:
        stop_words = set(stopwords.words("english"))
    except LookupError:
        stop_words = set()
    words = [word for word in text.split() if word not in stop_words]
    return " ".join(words)


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


def run_eda(frame):
    print("\nNull counts before cleaning:")
    print(frame.isnull().sum())

    print("\nDescribe:")
    print(frame.describe(include="all"))

    categorical_columns = frame.select_dtypes(include=["object", "category"]).columns
    for column in categorical_columns:
        print(f"\nTop value counts for {column}:")
        print(frame[column].value_counts(dropna=False).head(10))

    rating_column = find_column(frame.columns, ["rating", "score", "stars"])
    if rating_column is not None:
        target = frame[rating_column].apply(sentiment_from_rating)
        print("\nGroupby target sentiment:")
        print(frame.assign(sentiment=target).groupby("sentiment")[rating_column].describe())

    # Observation 1: The target is created from ratings: 1-2 Negative, 3 Neutral, 4-5 Positive.
    # Observation 2: Review text is the main feature, so text cleaning is required before vectorizing.
    # Observation 3: Missing review text is filled first, then empty cleaned rows are removed.
    # Observation 4: Rating values are numeric and can be median-filled before creating the target.
    # Observation 5: Class balance is checked with value_counts because accuracy can hide imbalance.


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
    cleaned["review"] = cleaned["review"].fillna("Unknown")
    cleaned["rating"] = pd.to_numeric(cleaned["rating"], errors="coerce")
    rating_fill = cleaned["rating"].median()
    if pd.isna(rating_fill):
        rating_fill = 3
    cleaned["rating"] = cleaned["rating"].fillna(rating_fill)
    cleaned["cleaned_review"] = cleaned["review"].apply(clean_text)
    cleaned["sentiment"] = cleaned["rating"].apply(sentiment_from_rating)
    cleaned = cleaned[cleaned["cleaned_review"].str.len() > 0]
    cleaned = cleaned[cleaned["sentiment"] != "Unknown"].copy()
    cleaned["text_length"] = cleaned["review"].astype(str).str.len()
    return cleaned


def main():
    ensure_nltk_stopwords()
    raw_reviews = load_dataset()
    run_eda(raw_reviews)
    cleaned_reviews = clean_reviews(raw_reviews)

    print("\nAfter cleaning:")
    print(f"Shape: {cleaned_reviews.shape}")
    print("\nNull counts:")
    print(cleaned_reviews.isnull().sum())
    print(f"\nVerified 0 nulls after cleaning: {cleaned_reviews.isnull().sum().sum() == 0}")
    print("\nRating distribution:")
    print(cleaned_reviews["rating"].value_counts().sort_index())
    print("\nSentiment distribution:")
    print(cleaned_reviews["sentiment"].value_counts())
    print("\nAverage rating by sentiment:")
    print(cleaned_reviews.groupby("sentiment")["rating"].mean())
    print("\nCleaned preview:")
    print(cleaned_reviews[["review", "rating", "sentiment", "cleaned_review"]].head())

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned_reviews.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved cleaned output to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
