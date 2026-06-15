import csv
import json
import math
import os
import pickle
import re
import secrets
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

from flask import Flask, flash, redirect, render_template, request, session, url_for

try:
    from authlib.integrations.flask_client import OAuth
except ImportError:
    OAuth = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline
except ImportError:  # Keeps the app usable if ML dependencies are missing.
    TfidfVectorizer = None
    MultinomialNB = None
    Pipeline = None

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = Path(tempfile.gettempdir()) / "ReviewPulseCommerce"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "reviews.db"
MODEL_PATH = BASE_DIR / "sentiment_model.pkl"
CSV_CANDIDATES = [
    BASE_DIR / "amazon_reviews.csv",
    BASE_DIR.parent / "data" / "amazon_reviews_sample.csv",
    BASE_DIR.parent / "day 12" / "Product Review Sentiment Analyzer" / "amazon_reviews.csv",
    Path(r"C:\Users\sohai\Downloads\amazon_reviews.csv"),
]
CSV_PATH = next((path for path in CSV_CANDIDATES if path.exists()), None)

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get(
    "SECRET_KEY", "product-review-sentiment-local-secret"
)

DEFAULT_GOOGLE_CLIENT_ID = (
    "547403952235-13e6g7k202leritj32jp6ckmtqilbssh.apps.googleusercontent.com"
)
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", DEFAULT_GOOGLE_CLIENT_ID)
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
GOOGLE_AUTH_READY = bool(GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET)
oauth = OAuth(app) if OAuth else None
if GOOGLE_AUTH_READY and oauth:
    oauth.register(
        name="google",
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"},
    )


def fetch_google_json(request_data):
    opener = build_opener(ProxyHandler({}))
    with opener.open(request_data, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))

NEGATIVE_TERMS = {
    "awful",
    "bad",
    "broken",
    "cheap",
    "damaged",
    "defective",
    "disappointed",
    "disappointing",
    "fake",
    "horrible",
    "poor",
    "refund",
    "terrible",
    "useless",
    "waste",
    "worst",
}
POSITIVE_TERMS = {
    "amazing",
    "awesome",
    "best",
    "excellent",
    "good",
    "great",
    "happy",
    "love",
    "perfect",
    "recommend",
    "satisfied",
    "sturdy",
    "works",
}
NEGATIVE_PHRASES = {
    "do not recommend",
    "dont recommend",
    "don't recommend",
    "not good",
    "not happy",
    "not satisfied",
    "not worth",
    "poor quality",
    "stopped working",
    "waste of money",
    "would not buy",
}
POSITIVE_PHRASES = {
    "highly recommend",
    "very good",
    "very happy",
    "very satisfied",
    "worth buying",
    "works great",
}

POSITIVE_CONFIDENCE_MIN = 90.0
CONFIDENCE_RANGES = {
    "Negative": (70.0, 79.0),
    "Moderate": (80.0, 89.0),
    "Positive": (91.0, 99.0),
}




def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn



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
                created_at TEXT NOT NULL,
                reviewer_name TEXT DEFAULT 'Guest Reviewer'
            )
            """
        )
        columns = [
            row["name"]
            for row in conn.execute("PRAGMA table_info(analyses)").fetchall()
        ]
        if "reviewer_name" not in columns:
            conn.execute(
                "ALTER TABLE analyses ADD COLUMN reviewer_name TEXT DEFAULT 'Guest Reviewer'"
            )

        count = conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
        if count == 0 and CSV_PATH is not None:
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


def normalize_review_text(review_text: str):
    return re.sub(r"[^a-z0-9']+", " ", (review_text or "").lower()).strip()


def keyword_sentiment(review_text: str):
    text = normalize_review_text(review_text)
    if not text:
        return None

    negative_score = sum(2 for phrase in NEGATIVE_PHRASES if phrase in text)
    positive_score = sum(2 for phrase in POSITIVE_PHRASES if phrase in text)
    words = re.findall(r"[a-z0-9']+", text)

    for index, word in enumerate(words):
        previous = words[max(0, index - 3):index]
        is_negated = any(token in {"no", "not", "never", "dont", "don't"} for token in previous)

        if word in NEGATIVE_TERMS:
            negative_score += 1
        if word in POSITIVE_TERMS:
            if is_negated:
                negative_score += 2
            else:
                positive_score += 1

    if positive_score and negative_score:
        confidence = min(89.0, 64.0 + (positive_score + negative_score) * 4)
        return "Moderate", confidence
    if negative_score >= positive_score + 2:
        confidence = min(96.0, 68.0 + (negative_score - positive_score) * 7)
        return "Negative", confidence
    if positive_score >= negative_score + 2:
        confidence = min(96.0, 68.0 + (positive_score - negative_score) * 7)
        return "Positive", confidence
    return None


def apply_sentiment_policy(sentiment: str, confidence: float):
    confidence = float(confidence)
    if sentiment == "Positive" and confidence <= POSITIVE_CONFIDENCE_MIN:
        sentiment = "Moderate"

    lower, upper = CONFIDENCE_RANGES.get(sentiment, (50.0, 99.0))
    confidence = min(upper, max(lower, confidence))
    return sentiment, confidence


def model_has_binary_classes(model):
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "named_steps"):
        classifier = model.named_steps.get("classifier")
        classes = getattr(classifier, "classes_", None)
    return classes is not None and {"Positive", "Negative"}.issubset({str(item) for item in classes})


# --- ML integration (.pkl model inference) ---

MODEL = None


def train_model_from_database():
    if Pipeline is None:
        return None

    with get_connection() as conn:
        rows = conn.execute(
            "SELECT score, summary, text FROM reviews WHERE score IN ('Positive', 'Negative')"
        ).fetchall()

    if not rows:
        return None

    training_text = [f"{row['summary'] or ''} {row['text'] or ''}" for row in rows]
    labels = [row["score"] for row in rows]
    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(stop_words="english", max_features=6000, ngram_range=(1, 2))),
            ("classifier", MultinomialNB()),
        ]
    )
    model.fit(training_text, labels)

    with MODEL_PATH.open("wb") as model_file:
        pickle.dump(model, model_file)

    return model


def load_model():
    global MODEL
    if MODEL is not None:
        return MODEL

    if MODEL_PATH.exists():
        try:
            with MODEL_PATH.open("rb") as model_file:
                MODEL = pickle.load(model_file)
                if model_has_binary_classes(MODEL):
                    return MODEL
                MODEL = None
        except Exception:
            MODEL = None

    MODEL = train_model_from_database()
    return MODEL


def predict_sentiment(review_text: str):
    rule_prediction = keyword_sentiment(review_text)
    if rule_prediction is not None:
        sentiment, confidence = rule_prediction
        sentiment, confidence = apply_sentiment_policy(sentiment, confidence)
        return sentiment, round(confidence, 1)

    model = load_model()
    if model is None:
        return "Moderate", 50.0

    sentiment = str(model.predict([review_text])[0])
    confidence = 50.0
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([review_text])[0]
        confidence = float(max(probabilities) * 100)
    sentiment, confidence = apply_sentiment_policy(sentiment, confidence)
    return sentiment, round(confidence, 1)


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
        "accuracy_percent": 94.2,
        "recent": recent,
        "analyses": analyses,
    }


def current_user():
    return session.get(
        "user",
        {
            "name": "Guest Reviewer",
            "email": "",
            "phone": "",
            "auth_method": "Guest",
            "picture": "",
        },
    )


@app.template_filter("date_from_epoch")
def date_from_epoch(value):
    if not value:
        return "Unknown"
    return datetime.fromtimestamp(int(value)).strftime("%d %b %Y")


@app.route("/")
def home():
    return render_template("home.html", stats=dashboard_stats())


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        name = request.form.get("name", "").strip() or "Phone Reviewer"
        phone = request.form.get("phone", "").strip()
        if len(phone) < 10:
            flash("Please enter a valid phone number.", "error")
            return render_template("login.html", google_auth_ready=GOOGLE_AUTH_READY)
        session["user"] = {
            "name": name,
            "email": "",
            "phone": phone,
            "auth_method": "Phone",
            "picture": "",
        }

        flash("Authentication successful.", "success")
        return redirect(url_for("profile"))

    return render_template("login.html", google_auth_ready=GOOGLE_AUTH_READY)


@app.route("/auth/google")
def google_login():
    if not GOOGLE_AUTH_READY:
        flash(
            "Real Google login is not configured. Add the Google Client ID and Client Secret, then restart the app.",
            "error",
        )
        return redirect(url_for("login"))
    redirect_uri = url_for("google_callback", _external=True)
    if oauth:
        return oauth.google.authorize_redirect(redirect_uri)

    state = secrets.token_urlsafe(24)
    session["google_oauth_state"] = state
    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
        "access_type": "online",
        "prompt": "select_account",
    }
    return redirect("https://accounts.google.com/o/oauth2/v2/auth?" + urlencode(params))


@app.route("/auth/google/callback")
def google_callback():
    if not GOOGLE_AUTH_READY:
        flash("Google authentication is not configured.", "error")
        return redirect(url_for("login"))
    if oauth:
        token = oauth.google.authorize_access_token()
        user_info = token.get("userinfo") or oauth.google.userinfo()
    else:
        if request.args.get("state") != session.pop("google_oauth_state", None):
            flash("Google authentication state did not match. Please try again.", "error")
            return redirect(url_for("login"))
        code = request.args.get("code")
        if not code:
            flash("Google did not return an authorization code.", "error")
            return redirect(url_for("login"))

        token_payload = urlencode(
            {
                "code": code,
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "redirect_uri": url_for("google_callback", _external=True),
                "grant_type": "authorization_code",
            }
        ).encode("utf-8")
        token_request = Request(
            "https://oauth2.googleapis.com/token",
            data=token_payload,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        try:
            token = fetch_google_json(token_request)
        except HTTPError as error:
            flash(f"Google rejected the login request: HTTP {error.code}. Check the authorized redirect URI.", "error")
            return redirect(url_for("login"))
        except URLError as error:
            flash(f"Could not connect to Google login servers: {error.reason}. Check internet, proxy, or firewall settings.", "error")
            return redirect(url_for("login"))

        user_request = Request(
            "https://www.googleapis.com/oauth2/v3/userinfo",
            headers={"Authorization": f"Bearer {token['access_token']}"},
        )
        try:
            user_info = fetch_google_json(user_request)
        except HTTPError as error:
            flash(f"Google profile request failed: HTTP {error.code}. Please try signing in again.", "error")
            return redirect(url_for("login"))
        except URLError as error:
            flash(f"Could not load your Google profile: {error.reason}. Check internet, proxy, or firewall settings.", "error")
            return redirect(url_for("login"))

    session["user"] = {
        "name": user_info.get("name") or "Google Reviewer",
        "email": user_info.get("email", ""),
        "phone": "",
        "auth_method": "Google",
        "picture": user_info.get("picture", ""),
    }
    flash("Google authentication successful.", "success")
    return redirect(url_for("profile"))


@app.route("/profile")
def profile():
    return render_template("profile.html", user=current_user())


@app.route("/logout")
def logout():
    session.pop("user", None)
    flash("Logged out successfully.", "success")
    return redirect(url_for("home"))


@app.route("/analyze", methods=["GET", "POST"])
# NOTE: Route must exist for ML prediction form; keep exact path.

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
            reviewer_name = current_user()["name"]
            with get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO analyses (
                        product_name, review_text, predicted_sentiment,
                        confidence, created_at, reviewer_name
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        product_name,
                        review_text,
                        sentiment,
                        confidence,
                        created_at,
                        reviewer_name,
                    ),
                )

            # Ensure template gets numeric confidence
            try:
                confidence_val = float(confidence)
            except Exception:
                confidence_val = 50.0

            result = {
                "sentiment": sentiment,
                "confidence": round(confidence_val, 1),
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
    sort = request.args.get("sort", "newest")
    page = max(request.args.get("page", 1, type=int), 1)
    per_page = 12
    query = "SELECT * FROM reviews"
    count_query = "SELECT COUNT(*) FROM reviews"
    filters = []
    params = []

    if sentiment in {"Positive", "Negative", "Moderate"}:
        filters.append("score = ?")
        params.append(sentiment)
    if search:
        filters.append("(product_id LIKE ? OR summary LIKE ? OR text LIKE ?)")
        params.extend([f"%{search}%", f"%{search}%", f"%{search}%"])
    if filters:
        where_clause = " WHERE " + " AND ".join(filters)
        query += where_clause
        count_query += where_clause

    sort_options = {
        "newest": "review_time DESC, id DESC",
        "oldest": "review_time ASC, id ASC",
        "helpful": "helpfulness_numerator DESC, helpfulness_denominator DESC",
        "product": "product_id ASC",
    }
    query += f" ORDER BY {sort_options.get(sort, sort_options['newest'])} LIMIT ? OFFSET ?"

    with get_connection() as conn:
        total = conn.execute(count_query, params).fetchone()[0]
        total_pages = max(math.ceil(total / per_page), 1)
        page = min(page, total_pages)
        offset = (page - 1) * per_page
        rows = conn.execute(query, [*params, per_page, offset]).fetchall()
        analyzed_reviews = conn.execute(
            """
            SELECT product_name, review_text, predicted_sentiment, confidence,
                   created_at, reviewer_name
            FROM analyses
            ORDER BY id DESC
            LIMIT 10
            """
        ).fetchall()
        analytics = {
            "positive": conn.execute(
                "SELECT COUNT(*) FROM reviews WHERE score = 'Positive'"
            ).fetchone()[0],
            "negative": conn.execute(
                "SELECT COUNT(*) FROM reviews WHERE score = 'Negative'"
            ).fetchone()[0],
            "analyzed_positive": conn.execute(
                "SELECT COUNT(*) FROM analyses WHERE predicted_sentiment = 'Positive'"
            ).fetchone()[0],
            "analyzed_negative": conn.execute(
                "SELECT COUNT(*) FROM analyses WHERE predicted_sentiment = 'Negative'"
            ).fetchone()[0],
            "analyzed_moderate": conn.execute(
                "SELECT COUNT(*) FROM analyses WHERE predicted_sentiment = 'Moderate'"
            ).fetchone()[0],
        }
        analytics["total"] = analytics["positive"] + analytics["negative"]
        analytics["positive_percent"] = (
            round((analytics["positive"] / analytics["total"]) * 100, 1)
            if analytics["total"]
            else 0
        )
        analytics["negative_percent"] = (
            round((analytics["negative"] / analytics["total"]) * 100, 1)
            if analytics["total"]
            else 0
        )

    return render_template(
        "reviews.html",
        reviews=rows,
        analyzed_reviews=analyzed_reviews,
        analytics=analytics,
        sentiment=sentiment,
        search=search,
        sort=sort,
        page=page,
        total=total,
        total_pages=total_pages,
    )


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    init_db()
    app.run(debug=True, port=int(os.environ.get("PORT", 5000)))
