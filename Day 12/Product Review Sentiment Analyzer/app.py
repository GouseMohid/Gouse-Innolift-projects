import csv
import math
import os
import re
import sqlite3
import tempfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from flask import Flask, flash, redirect, render_template, request, url_for


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("LOCALAPPDATA", tempfile.gettempdir())) / "ReviewPulseCommerce"
try:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
except PermissionError:
    DATA_DIR = Path(tempfile.gettempdir()) / "ReviewPulseCommerce"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "reviews.db"
LOCAL_CSV_PATH = BASE_DIR / "amazon_reviews.csv"
DOWNLOADS_CSV_PATH = Path(r"C:\Users\sohai\Downloads\amazon_reviews.csv")
CSV_PATH = LOCAL_CSV_PATH if LOCAL_CSV_PATH.exists() else DOWNLOADS_CSV_PATH

app = Flask(__name__)
app.config["SECRET_KEY"] = "product-review-sentiment-demo"

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z']+")
STOP_WORDS = {
    "the", "and", "for", "with", "this", "that", "was", "were", "are", "you",
    "have", "has", "had", "but", "not", "from", "they", "will", "would",
    "there", "their", "about", "into", "your", "just", "what", "when", "where",
    "which", "than", "then", "them", "very", "more", "some", "been", "also",
}


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def tokenize(text):
    return [
        token.lower()
        for token in TOKEN_RE.findall(text or "")
        if len(token) > 2 and token.lower() not in STOP_WORDS
    ]


def init_db():
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY,
                product_id TEXT NOT NULL,
                user_id TEXT,
                profile_name TEXT,
                helpfulness_numerator INTEGER DEFAULT 0,
                helpfulness_denominator INTEGER DEFAULT 0,
                score TEXT NOT NULL,
                review_time INTEGER,
                summary TEXT,
                text TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_name TEXT,
                review_text TEXT NOT NULL,
                predicted_sentiment TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )

        count = conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
        if count == 0 and CSV_PATH.exists():
            seed_reviews(conn)


def seed_reviews(conn):
    with CSV_PATH.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = []
        for row in reader:
            if row.get("Score") not in {"Positive", "Negative"}:
                continue
            rows.append(
                (
                    int(row["Id"]),
                    row.get("ProductId", ""),
                    row.get("UserId", ""),
                    row.get("ProfileName", ""),
                    int(row.get("HelpfulnessNumerator") or 0),
                    int(row.get("HelpfulnessDenominator") or 0),
                    row["Score"],
                    int(row.get("Time") or 0),
                    row.get("Summary", ""),
                    row.get("Text", ""),
                )
            )

        conn.executemany(
            """
            INSERT OR IGNORE INTO reviews (
                id, product_id, user_id, profile_name, helpfulness_numerator,
                helpfulness_denominator, score, review_time, summary, text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )


def train_sentiment_model():
    word_counts = {"Positive": Counter(), "Negative": Counter()}
    class_counts = Counter()

    with get_connection() as conn:
        rows = conn.execute("SELECT score, summary, text FROM reviews").fetchall()

    for row in rows:
        label = row["score"]
        class_counts[label] += 1
        word_counts[label].update(tokenize(f"{row['summary']} {row['text']}"))

    vocabulary = set(word_counts["Positive"]) | set(word_counts["Negative"])
    return word_counts, class_counts, max(len(vocabulary), 1)


def predict_sentiment(review_text):
    word_counts, class_counts, vocabulary_size = train_sentiment_model()
    tokens = tokenize(review_text)
    total_reviews = sum(class_counts.values()) or 1
    scores = {}

    for label in ("Positive", "Negative"):
        total_words = sum(word_counts[label].values())
        log_score = math.log((class_counts[label] or 1) / total_reviews)
        for token in tokens:
            log_score += math.log(
                (word_counts[label][token] + 1) / (total_words + vocabulary_size)
            )
        scores[label] = log_score

    positive = math.exp(scores["Positive"] - max(scores.values()))
    negative = math.exp(scores["Negative"] - max(scores.values()))
    confidence = max(positive, negative) / (positive + negative)
    sentiment = "Positive" if scores["Positive"] >= scores["Negative"] else "Negative"
    return sentiment, round(confidence * 100, 1)


def dashboard_stats():
    with get_connection() as conn:
        total = conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
        positive = conn.execute(
            "SELECT COUNT(*) FROM reviews WHERE score = 'Positive'"
        ).fetchone()[0]
        negative = conn.execute(
            "SELECT COUNT(*) FROM reviews WHERE score = 'Negative'"
        ).fetchone()[0]
        recent = conn.execute(
            """
            SELECT product_id, profile_name, score, summary, text,
                   helpfulness_numerator, helpfulness_denominator, review_time
            FROM reviews
            ORDER BY id DESC
            LIMIT 12
            """
        ).fetchall()
        analyses = conn.execute(
            "SELECT * FROM analyses ORDER BY id DESC LIMIT 8"
        ).fetchall()

    return {
        "total": total,
        "positive": positive,
        "negative": negative,
        "positive_percent": round((positive / total) * 100, 1) if total else 0,
        "negative_percent": round((negative / total) * 100, 1) if total else 0,
        "recent": recent,
        "analyses": analyses,
    }


@app.template_filter("date_from_epoch")
def date_from_epoch(value):
    if not value:
        return "Unknown"
    return datetime.fromtimestamp(int(value)).strftime("%d %b %Y")


@app.route("/")
def home():
    return render_template("home.html", stats=dashboard_stats())


@app.route("/analyze", methods=["GET", "POST"])
def analyze():
    result = None
    product_name = ""
    review_text = ""

    if request.method == "POST":
        product_name = request.form.get("product_name", "").strip()
        review_text = request.form.get("review_text", "").strip()
        if len(review_text) < 10:
            flash("Please enter a review with at least 10 characters.", "error")
        else:
            sentiment, confidence = predict_sentiment(review_text)
            created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO analyses (
                        product_name, review_text, predicted_sentiment,
                        confidence, created_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (product_name, review_text, sentiment, confidence, created_at),
                )
            result = {
                "sentiment": sentiment,
                "confidence": confidence,
                "product_name": product_name or "Untitled product",
            }
            flash("Review analyzed successfully.", "success")

    return render_template(
        "analyze.html",
        result=result,
        product_name=product_name,
        review_text=review_text,
    )


@app.route("/reviews")
def reviews():
    sentiment = request.args.get("sentiment", "All")
    search = request.args.get("search", "").strip()
    query = "SELECT * FROM reviews"
    filters = []
    params = []

    if sentiment in {"Positive", "Negative"}:
        filters.append("score = ?")
        params.append(sentiment)
    if search:
        filters.append("(product_id LIKE ? OR summary LIKE ? OR text LIKE ?)")
        params.extend([f"%{search}%", f"%{search}%", f"%{search}%"])
    if filters:
        query += " WHERE " + " AND ".join(filters)
    query += " ORDER BY id DESC LIMIT 100"

    with get_connection() as conn:
        rows = conn.execute(query, params).fetchall()

    return render_template(
        "reviews.html",
        reviews=rows,
        sentiment=sentiment,
        search=search,
    )


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    init_db()
    app.run(debug=True, port=int(os.environ.get("PORT", 5000)))
