from pathlib import Path
import pickle
import re
import string

import numpy as np
from nltk.corpus import stopwords


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "model.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.pkl"

CASES = [
    "The product arrived early, feels sturdy, and works perfectly every day.",
    "It is okay for the price, but the setup was confusing and the box looked old.",
    "Terrible quality. It stopped working after two days and customer support ignored me.",
]


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


def load_artifacts():
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        raise FileNotFoundError("Run `python train.py` first to create model.pkl and tfidf_vectorizer.pkl.")

    with MODEL_PATH.open("rb") as model_file:
        model = pickle.load(model_file)
    with VECTORIZER_PATH.open("rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer


def predict_review(review, model, vectorizer):
    cleaned = clean_text(review)
    vector = vectorizer.transform([cleaned])
    sentiment = model.predict(vector)[0]
    probabilities = model.predict_proba(vector)[0]
    confidence = float(np.max(probabilities) * 100)
    class_confidence = {
        label: round(float(probability) * 100, 2)
        for label, probability in zip(model.classes_, probabilities)
    }
    return sentiment, round(confidence, 2), class_confidence


def main():
    model, vectorizer = load_artifacts()
    for index, review in enumerate(CASES, start=1):
        sentiment, confidence, class_confidence = predict_review(review, model, vectorizer)
        print(f"Case {index}")
        print(f"Review: {review}")
        print(f"Prediction: {sentiment} ({confidence}%)")
        print(f"Class probabilities: {class_confidence}")
        print()


if __name__ == "__main__":
    main()
