import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from sentiment_model import clean_text


df = pd.read_csv("data/amazon_reviews_sample.csv")

df["sentiment"] = "Neutral"
df.loc[df["rating"] >= 4, "sentiment"] = "Positive"
df.loc[df["rating"] <= 2, "sentiment"] = "Negative"
df["cleaned_review"] = df["review"].apply(clean_text)

test_sizes = [0.1, 0.2, 0.3]
results = []

print("Train-Test Split Explorer")
print("-------------------------")

for test_size in test_sizes:
    x_train, x_test, y_train, y_test = train_test_split(
        df["cleaned_review"],
        df["sentiment"],
        test_size=test_size,
        random_state=42,
    )

    vectorizer = TfidfVectorizer()
    x_train_tfidf = vectorizer.fit_transform(x_train)
    x_test_tfidf = vectorizer.transform(x_test)

    model = MultinomialNB()
    model.fit(x_train_tfidf, y_train)

    predictions = model.predict(x_test_tfidf)
    accuracy = accuracy_score(y_test, predictions)

    results.append(
        {
            "test_size": test_size,
            "train_size": len(x_train),
            "test_size_count": len(x_test),
            "accuracy": accuracy,
        }
    )

    print(f"\nTest size value: {test_size}")
    print(f"Train size: {len(x_train)}")
    print(f"Test size: {len(x_test)}")
    print(f"Model accuracy: {accuracy:.2f}")

best_result = max(results, key=lambda result: result["accuracy"])

print("\nBest split:")
print(f"Test size value: {best_result['test_size']}")
print(f"Train size: {best_result['train_size']}")
print(f"Test size: {best_result['test_size_count']}")
print(f"Best accuracy: {best_result['accuracy']:.2f}")
